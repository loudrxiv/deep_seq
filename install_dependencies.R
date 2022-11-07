# ----- Install Bioconductor -----
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()

# ----- Install Newest Reticulate -----
devtools::install_github("rstudio/reticulate")

# ----- Renv -----
# manages packages, which is nice...see
# https://rstudio.github.io/renv/articles/renv.html
install.packages("renv")

# ----- Install 'BSgenome.Hsapiens.UCSC.hg*' -----
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")

# ---- Install tfdatasets for scalable ML -----
install.packages("https://cran.r-project.org/src/contrib/Archive/rlang/rlang_1.0.1.tar.gz", repo=NULL, type="source")
devtools::install_github("rstudio/tfdatasets")

# ---- Load files directly into environment ----
install.packages("remotes")
remotes::install_github("konradedgar/KEmisc")

# ---- Install TensorHub ----
install.packages("tfhub")

# tfaddons====
devtools::install_github('henry090/tfaddons')
tfaddons::install_tfaddons()