args <- commandArgs(trailingOnly = TRUE)

target <- args[1]
csv_file_path <- args[2]
number_of_predictors <- args[3]

library("varrank")

dataset <- read.csv(csv_file_path)

# Cannot be Null
#discretization_methods <- list("NULL", "fd", "cencov", "rice", "terrell-scott", "sturges", "scott", "kmeans")
#methods <- list("battiti", "kwak", "peng", "esteves")
discretization_methods <- list("cencov") #, "rice", "terrell-scott", "sturges", "scott")
methods <- list("esteves")
algorithms <- list("forward")
print("#######################################################################################")
print("#######################################################################################")
print("#######################################################################################")
print(paste(c("          Selecting features with VARRANK for: "), target))
print("#######################################################################################")
print("#######################################################################################")
print("#######################################################################################")
for (disc_method in discretization_methods) {
    for (method in methods) {
        for (algo in algorithms) {
            print("=======================================================================================")
            print("=======================================================================================")
            print("=======================================================================================")
            print(paste(c("Trying  method: ", method)))
            print(paste(c("Trying discretization method: ", disc_method)))
            print(paste(c("Trying algorithm: ", algo)))
            varrank.output <- varrank(
                data.df = dataset,
                method = method,
                variable.important = target,
                discretization.method = disc_method,
                algorithm = algo,
                scheme="mid",
                verbose=TRUE,
                n.var= as.numeric(number_of_predictors)
            )
            print("Result:")
            print(varrank.output)
            write.csv(varrank.output[1], file=csv_file_path, row.names=FALSE)
            print("")
        }
    }
}
