# R/setup.R — Day 1 R environment bootstrap
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}
renv::init(bare = TRUE, restart = FALSE)
pkgs <- c(
  "terra", "sf", "spdep", "blockCV", "tmap",
  "pROC", "pwr", "effectsize", "Kendall", "trend", "mblm", "boot", "lme4", "ggeffects",
  "caret", "keras", "tensorflow",
  "ggplot2", "scales", "patchwork", "viridis", "RColorBrewer",
  "kableExtra", "knitr", "rmarkdown"
)
install.packages(pkgs, repos = "https://cloud.r-project.org",
                 dependencies = TRUE, Ncpus = 4)
renv::snapshot(prompt = FALSE)
cat("\nrenv.lock written. R environment is reproducible.\n")
sessionInfo()
