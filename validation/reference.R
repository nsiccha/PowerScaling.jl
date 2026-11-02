args <- commandArgs(trailingOnly = TRUE)
out <- args[[1]]
library(priorsense)
library(posterior)
d <- read.csv(file.path(out, "draws.csv"))
distance <- cjs_dist(d$mu, d$mu, y_weights = d$julia_weights)
draws <- as_draws_matrix(d[, c("mu", "lprior", "log_lik")])
s <- powerscale_sensitivity(draws, variable = "mu", moment_match = FALSE)
writeLines(sprintf("%.17g", c(distance, s$prior[[1]], s$likelihood[[1]])),
           file.path(out, "r_values.txt"))
cat("R_VERSIONS", R.version.string,
    "priorsense=", as.character(packageVersion("priorsense")),
    "posterior=", as.character(packageVersion("posterior")),
    "loo=", as.character(packageVersion("loo")), "\n")
print(s)
