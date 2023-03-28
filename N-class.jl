using Gen, PyCall, GenTF, Distributions, Metrics, PyPlot, SyntheticDatasets, DataFrames, Random
@pyimport tensorflow as tf # Only works if you do ENV["PYTHON"] = "<python>", where <python> is an absolute path to a python installation wtih tensorflow included
                           # More at https://probcomp.github.io/GenTF/dev/#Installation-1

@gen function BNN_model(x::Matrix{Float64}, nn_model, n_weights::Integer)
    weights = Float32[] 
    
    for i = 1:n_weights
        push!(weights, {(:weights, i)} ~ normal(0, 1))
    end

    weights_L1 = [[weights[1] weights[2] weights[3] weights[4] weights[5] weights[6]; weights[7] weights[8] weights[9] weights[10] weights[11] weights[12]; weights[13] weights[14] weights[15] weights[16] weights[17] weights[18]], [weights[19], weights[20], weights[21], weights[22], weights[23], weights[24]]]
    weights_L2 = [[weights[25] weights[26] weights[27] weights[28] weights[29] weights[30]; weights[31] weights[32] weights[33] weights[34] weights[35] weights[36];
                   weights[37] weights[38] weights[39] weights[40] weights[41] weights[42]; weights[43] weights[44] weights[45] weights[46] weights[47] weights[48];
                   weights[49] weights[50] weights[51] weights[52] weights[53] weights[54]; weights[55] weights[56] weights[57] weights[58] weights[59] weights[60]],
                  [weights[61], weights[62], weights[63], weights[64], weights[65], weights[66]]]
    weights_L3 = [[weights[67] weights[68] weights[69] weights[70]; weights[71]  weights[72]  weights[73]  weights[74];
                   weights[75] weights[76] weights[77] weights[78]; weights[79]  weights[80]  weights[81]  weights[82];
                   weights[83] weights[84] weights[85] weights[86]; weights[87]  weights[88]  weights[89]  weights[90]], [weights[91], weights[92], weights[93], weights[94]]] # I don't know why the semicolons at the end are in there
    # Ideally you don't hardcode those but hey you gotta do what you gotta do

    nn_model[:get_layer](index=0)[:set_weights](weights_L1)
    nn_model[:get_layer](index=1)[:set_weights](weights_L2)
    nn_model[:get_layer](index=2)[:set_weights](weights_L3)

    y = nn_model[:predict](x, batch_size = 500)
    y = Float64.(y) # If left at Float32 it breaks and I don't know why
    final_y = Int64[]
    for i=1:size(y)[1]
        push!(final_y, {(:y, i)} ~ Gen.categorical(y[i, :]))
    end
    final_y
end

function make_constraints(y::Vector{Int64}) # Add the observed y's to the equation
    constraints = Gen.choicemap()
    for i = 1:size(y)[1]
        constraints[(:y, i)] = y[i]
    end
    constraints
end

function block_resimulation_update(trace, n_weights::Integer) # MCMC magic
    for i = 1:n_weights
      latent_variable = Gen.select((:weights, i))
      (trace, _) = mh(trace, latent_variable)
    end
  trace
end
  
function block_resimulation_inference(x::Matrix{Float64}, nn_model, n_weights::Integer, y::Vector{Int64}, n_burnin::Integer, n_samples::Integer)
    observations = make_constraints(y)
    (trace, _) = generate(BNN_model, (x, nn_model, n_weights, ), observations)
        
    for i = 1:n_burnin
        trace = block_resimulation_update(trace, n_weights)
    end

    traces = []
    for i = 1:n_samples
        trace = block_resimulation_update(trace, n_weights)
        push!(traces, trace)
        if mod(i,100) == 0
          println("Iteration $i")
        end
    end
    traces
end

model = tf.keras.Sequential()
model[:add](tf.keras.layers.Dense(6, input_shape=(3,), activation="relu"))
model[:add](tf.keras.layers.Dense(6, activation="relu"))
model[:add](tf.keras.layers.Dense(4, activation="softmax"))
model[:compile](loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])

df = SyntheticDatasets.make_classification(n_samples = 1000, n_features = 3, n_informative = 3, n_redundant = 0, n_classes=4, n_clusters_per_class = 1, class_sep=1.5)
df = df[shuffle(1:end), :]
Y = df[:, :label] .+ 1
println(first(df, 20))
println(last(df, 20))
df = DataFrames.select(df, Not(:label))
X = Matrix(df)

fig, ax = subplots()
ax = fig[:add_subplot](projection="3d")
ax[:scatter](X[:,1], X[:, 2], X[:,3], s = 10, c=Y)
ax[:set_xlabel]("x_1")
ax[:set_ylabel]("x_2")
ax[:set_ylabel]("x_3")
ax[:set_title]("N-class dataset")
plt[:show]()

n_weights = 94

traces = block_resimulation_inference(X[1:500, :], model, n_weights, Y[1:500], 100, 300)

predicted_labels = model[:predict](X[501:1000, :])
println(predicted_labels)
predicted_labels = [argmax(predicted_labels[i, :]) for i in 1:size(predicted_labels)[1]]
println(predicted_labels)
println(Y[501:1000])
predicted_labels = reshape(predicted_labels, (500,))

accuracy = sum(predicted_labels .== Y[501:1000])/size(predicted_labels)[1]
println("Accuracy is $accuracy")

freq_w1 = [traces[i][(:weights, 1)] for i=1:300]
fig, ax = subplots()
ax[:hist](freq_w1, bins=[-3, -2, -1, 0, 1, 2, 3], label="posterior of w11", color="#348ABD")
xlim(-3, 3)
legend()
plt[:show]()

freq_w2 = [traces[i][(:weights, 2)] for i=1:300]
fig, ax = subplots()
ax[:hist](freq_w2, bins=[-3, -2, -1, 0, 1, 2, 3], label="posterior of w12", color="#348ABD")
xlim(-3, 3)
legend()
plt[:show]()

freq_w3 = [traces[i][(:weights, 3)] for i=1:300]
fig, ax = subplots()
ax[:hist](freq_w3, bins=[-3, -2, -1, 0, 1, 2, 3], label="posterior of w13", color="#348ABD")
xlim(-3, 3)
legend()
plt[:show]()

wrong_labels = predicted_labels .== Y[501:1000]

fig, ax = subplots()
ax = fig[:add_subplot](projection="3d")
ax[:scatter](X[501:1000,1], X[501:1000, 2], X[501:1000, 3], s = 10, c=wrong_labels)
ax[:set_xlabel]("x_1")
ax[:set_ylabel]("x_2")
ax[:set_ylabel]("x_3")
ax[:set_title]("Incorrectly labelled")
plt[:show]()