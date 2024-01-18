if (!requireNamespace("randomForestSRC", lib='/mind_data/jeej/Rlib', quietly = TRUE)) { #<--replace with your Rlib location
  install.packages("randomForestSRC",lib='/mind_data/jeej/Rlib') #<--replace with your Rlib location
}

# following the example in https://www.randomforestsrc.org/articles/competing.html
# but substituting in the data from this manuscript (LB+ model)

# Load the required libraries
library(randomForestSRC,lib='/mind_data/jeej/Rlib') #<--replace with your Rlib location
library(data.table)
library(dplyr)

# discovery cohort (MSK-ACCESS)
vte <- read.csv("../data/discovery.csv", header=TRUE)
logical_columns <- sapply(vte, function(x) all(x %in% c("True", "False")))
vte <- vte %>%
  mutate_at(vars(names(logical_columns)[logical_columns]), ~ ifelse(. == "True", TRUE, FALSE))

# train rfsrc	
genes = read.csv("common_gene_list.txt", header=TRUE)
vte.obj <- rfsrc(Surv(stop, CAT_DEATH_ENDPT) ~ ., data=vte[c("Non.Small.Cell.Lung.Cancer", "Breast.Cancer", "Pancreatic.Cancer",
            "Melanoma", "Prostate.Cancer", "Bladder.Cancer",
            "Esophagogastric.Cancer", "Hepatobiliary.Cancer", "Colorectal.Cancer",
	    "X.ctDNA","log10.max.VAF.",
            "log10.cfDNA.concentration.","chemotherapy",genes$Gene,
	    "stop","CAT_DEATH_ENDPT")])

# summarize results
pdf("rsf_cmprsk_vte.pdf", width = 8, height = 8)
par(cex.axis = 2.0, cex.lab = 2.0, cex.main = 2.0, mar = c(6.0,6,1,1), mgp = c(4, 1, 0))
plot.competing.risk(vte.obj)
dev.off()
print(vte.obj)

# apply to validation cohort
vte2 <- read.csv("../data/validation.csv", header=TRUE)
logical_columns <- sapply(vte2, function(x) all(x %in% c("True", "False")))
vte2 <- vte2 %>%
  mutate_at(vars(names(logical_columns)[logical_columns]), ~ ifelse(. == "True", TRUE, FALSE))

predictions <- predict(vte.obj, vte2)
print(predictions)
write.csv(predictions$predicted,"vte_riskscores_validation_cmprsk.csv", row.names=TRUE)
