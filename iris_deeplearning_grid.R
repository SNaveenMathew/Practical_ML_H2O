source("initialize_iris.R")

grd <- h2o.grid("deeplearning", search_criteria = list(strategy = "RandomDiscrete",
                                                       max_models = 40),
                hyper_params = list(seed = 1, l1 = c(0, 1e-6, 3e-6, 1e-5),
                                    l2 = c(0, 1e-6, 3e-6, 1e-5),
                                    input_dropout_ratio = c(0, 0.1, 0.2, 0.3),
                                    hidden_dropout_ratios = list(
                                      c(0,0), c(0.2, 0.2), c(0.4, 0.4), c(0.6, 0.6)
                                    )),
                grid_id = "iris-dl",
                x = 1:4, y = 5, hidden = c(8, 8),
                epochs = 10, training_frame = train, validation_frame = valid,
                activation = "RectifierWithDropout")

grd

best_model <- h2o.getModel(grd@model_ids[[1]])

h2o.saveModel(best_model, "iris_best_dl_model.mdl")

h2o.performance(best_model, valid)
h2o.performance(best_model, test)
