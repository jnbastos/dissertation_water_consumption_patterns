setwd(dir = "/Users/jnb/UMinho/[Tese] Water Consumption/water_consumption_v2")

library(dplyr)

# TESTS
test_df <- read.csv(file='./tests/penguins.csv')
df[['bill_length_mm', 'flipper_length_mm']]

test_df <- select(test_df, bill_length_mm, flipper_length_mm)

na.omit(test_df)

test_df

# RAW
df_gdc <- read.csv(file='./Data/df_gdc.csv')
ts_gdc <- read.csv(file='./Data/ts_gdc.csv')

df_hdc <- read.csv(file='./Data/df_gdc.csv')
ts_hdc <- read.csv(file='./Data/ts_gdc.csv')

df_hmdc <- read.csv(file='./Data/df_gdc.csv')
ts_hmdc <- read.csv(file='./Data/ts_gdc.csv')

df_hwdc <- read.csv(file='./Data/df_gdc.csv')
ts_hwdc <- read.csv(file='./Data/ts_gdc.csv')

# NORMALIZED
df_gdc_norm <- read.csv(file='./Data/df_gdc_norm.csv')
ts_gdc_norm <- read.csv(file='./Data/ts_gdc_norm.csv')

df_hdc_norm <- read.csv(file='./Data/df_gdc_norm.csv')
ts_hdc_norm <- read.csv(file='./Data/ts_gdc_norm.csv')

df_hmdc_norm <- read.csv(file='./Data/df_gdc_norm.csv')
ts_hmdc_norm <- read.csv(file='./Data/ts_gdc_norm.csv')

df_hwdc_norm <- read.csv(file='./Data/df_gdc_norm.csv')
ts_hwdc_norm <- read.csv(file='./Data/ts_gdc_norm.csv')

summary(df_gdc)

set.seed(42)
clstr <- kmeans(test_df, 5, nstart=5)
library(factoextra)

# Elbow method
fviz_nbclust(df_gdc, kmeans, method="wss", nstart = 5)

# Silhouette method
fviz_nbclust(df_gdc, kmeans, method="silhouette", nstart = 5)

# Gap statistic method

png("./tests/gap.png")
fviz_nbclust(na.omit(test_df), kmeans, method="gap", k.max=10, nboot=100)
dev.off()

## RAW

# GDC
png("img/raw/GDC/Ap1_km_gap.png")
fviz_nbclust(ts_gdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/raw/GDC/Ap2_km_gap.png")
fviz_nbclust(df_gdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

# HDC
png("img/raw/HDC/Ap1_km_gap.png")
fviz_nbclust(ts_hdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/raw/HDC/Ap2_km_gap.png")
fviz_nbclust(df_hdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

# HMDC
png("img/raw/HMDC/Ap1_km_gap.png")
fviz_nbclust(ts_hmdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/raw/HMDC/Ap2_km_gap.png")
fviz_nbclust(df_hmdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

# HWDC
png("img/raw/HWDC/Ap1_km_gap.png")
fviz_nbclust(ts_hwdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/raw/HWDC/Ap2_km_gap.png")
fviz_nbclust(df_hwdc, kmeans, method="gap", k.max=6, nboot=500)
dev.off()


## NORMALIZED

# GDC
png("img/norm/GDC/Ap1_km_gap.png")
fviz_nbclust(ts_gdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/norm/GDC/Ap2_km_gap.png")
fviz_nbclust(df_gdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

# HDC
png("img/norm/HDC/Ap1_km_gap.png")
fviz_nbclust(ts_hdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/norm/HDC/Ap2_km_gap.png")
fviz_nbclust(df_hdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

# HMDC
png("img/norm/HMDC/Ap1_km_gap.png")
fviz_nbclust(ts_hmdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/norm/HMDC/Ap2_km_gap.png")
fviz_nbclust(df_hmdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

# HWDC
png("img/norm/HWDC/Ap1_km_gap.png")
fviz_nbclust(ts_hwdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()

png("img/norm/HWDC/Ap2_km_gap.png")
fviz_nbclust(df_hwdc_norm, kmeans, method="gap", k.max=6, nboot=500)
dev.off()


### Gap statistic
library(cluster)
set.seed(0)
# Compute gap statistic for kmeans
# Recommended value for B is ~500
gap_stat <- clusGap(df_gdc_norm, FUN = kmeans, K.max = 6, B = 500)
print(gap_stat, method = "firstmax")
gc()
fviz_gap_stat(gap_stat)



##
library(cluster)
sil = silhouette(clstr$cluster, dist(df_gdc))
fviz_silhouette(sil)


#install.packages("data.table", type = "binary")
gap_stat <- clusGap(df_gdc, FUN = kmeans, nstart = 5, K.max = 10, B = 500)
print(gap_stat, method = "firstmax")
fviz_gap_stat(gap_stat)

