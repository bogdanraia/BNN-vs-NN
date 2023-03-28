using Gen, PyCall, GenTF, Distributions, Metrics, PyPlot, SyntheticDatasets, DataFrames, Random
@pyimport tensorflow as tf # Only works if you do ENV["PYTHON"] = "<python>", where <python> is an absolute path to a python installation wtih tensorflow included
                           # More at https://probcomp.github.io/GenTF/dev/#Installation-1

@gen function BNN_model(x::Matrix{Float64}, nn_model, n_weights::Integer)
    weights = Float32[] 
    
    for i = 1:n_weights
        push!(weights, {(:weights, i)} ~ normal(0, 1))
    end

    weights_L1 = [[weights[1] weights[2] weights[3] weights[4]; weights[5] weights[6] weights[7] weights[8]], [weights[9], weights[10], weights[11], weights[12]]]
    weights_L2 = [[weights[13] weights[14] weights[15] weights[16]; weights[17] weights[18] weights[19] weights[20]; weights[21] weights[22] weights[23] weights[24]; weights[25] weights[26] weights[27] weights[28]], [weights[29], weights[30], weights[31], weights[32]]]
    weights_L3 = [[weights[33]; weights[34]; weights[35]; weights[36];;], [weights[37]]] # I don't know why the semicolons at the end are in there
    # Ideally you don't hardcode those but hey you gotta do what you gotta do

    nn_model[:get_layer](index=0)[:set_weights](weights_L1)
    nn_model[:get_layer](index=1)[:set_weights](weights_L2)
    nn_model[:get_layer](index=2)[:set_weights](weights_L3)

    y = nn_model[:predict](x, batch_size = 500)
    y = Float64.(y) # If left at Float32 it breaks and I don't know why
    final_y = Bool[]
    for i=1:size(y)[1]
        push!(final_y, {(:y, i)} ~ Gen.bernoulli(y[i]))
    end
    final_y
end

function make_constraints(y::BitVector) # Add the observed y's to the equation
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
  
function block_resimulation_inference(x::Matrix{Float64}, nn_model, n_weights::Integer, y::BitVector, n_burnin::Integer, n_samples::Integer)
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
model[:add](tf.keras.layers.Dense(4, input_shape=(2,), activation="relu"))
model[:add](tf.keras.layers.Dense(4, activation="relu"))
model[:add](tf.keras.layers.Dense(1, activation="sigmoid"))
model[:compile](loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])

df = SyntheticDatasets.make_circles(n_samples = 1000, shuffle=true, noise = 0.03)
df = df[shuffle(1:end), :]
Y = df[:, :label] 
println(first(df, 20))
println(last(df, 20))
df = DataFrames.select(df, Not(:label))
X = Matrix(df)

Y = Y .== ones(Int64, 1000) # I chose BitVector, therefore I shall suffer 💀 (adhoc Y BitVector convert)

fig, ax = subplots()
ax[:scatter](X[:,1], X[:, 2], s = 10, c=Y)
ax[:set_xlabel]("x_1")
ax[:set_ylabel]("x_2")
ax[:set_title]("The Circle Dataset")
plt[:show]()

n_weights = 37

traces = block_resimulation_inference(X[1:500, :], model, n_weights, Y[1:500], 1000, 5000) # Sometimes it fails to learn inner circle and gives poor acc

predicted_labels = model[:predict](X[501:1000, :])
predicted_labels = reshape(predicted_labels, (500,))

accuracy = binary_accuracy(predicted_labels, Y[501:1000])
println("Accuracy is $accuracy")

freq_w1 = [traces[i][(:weights, 1)] for i=1:3000]
fig, ax = subplots()
ax[:hist](freq_w1, bins=[-3, -2, -1, 0, 1, 2, 3], label="posterior of w11", color="#348ABD")
xlim(-3, 3)
legend()
plt[:show]()

freq_w2 = [traces[i][(:weights, 2)] for i=1:3000]
fig, ax = subplots()
ax[:hist](freq_w2, bins=[-3, -2, -1, 0, 1, 2, 3], label="posterior of w12", color="#348ABD")
xlim(-3, 3)
legend()
plt[:show]()

freq_w3 = [traces[i][(:weights, 3)] for i=1:3000]
fig, ax = subplots()
ax[:hist](freq_w3, bins=[-3, -2, -1, 0, 1, 2, 3], label="posterior of w13", color="#348ABD")
xlim(-3, 3)
legend()
plt[:show]()

predicted_labels = predicted_labels .> 0.5
wrong_labels = predicted_labels .== Y[501:1000]

fig, ax = subplots()
ax[:scatter](X[501:1000,1], X[501:1000, 2], s = 10, c=wrong_labels)
ax[:set_xlabel]("x_1")
ax[:set_ylabel]("x_2")
ax[:set_title]("Incorrectly labelled")
plt[:show]()