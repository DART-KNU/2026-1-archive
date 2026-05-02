install.packages("SimMultiCorrData")
install.packages("PDQutils")

library(SimMultiCorrData)
library(PDQutils)
library(quantmod)

# Data
getSymbols("^DJI", from = "1990-01-03", to = Sys.Date(), src = "yahoo")

stock <- Ad(DJI)
stk.rtn <- diff(log(stock))

returns <- as.vector(stk.rtn)
returns <- returns[!is.na(returns)]

# Empirical density
stk.dens <- density(returns, na.rm = TRUE)

# Moments: mean, sd, skewness, kurtosis
m <- calc_moments(returns)

x <- stk.dens$x

# Gaussian density
qx <- dnorm(x, mean = m[1], sd = m[2])

# Plot
dev.new(width = 16, height = 8)

plot(
  x,
  stk.dens$y,
  type = "l",
  lwd = 4,
  col = "red",
  xlab = "x",
  ylab = "",
  main = "Empirical Density vs Edgeworth Expansion",
  xlim = c(-0.1, 0.1),
  ylim = c(0, 65),
  xaxs = "i",
  yaxs = "i",
  las = 1,
  cex.lab = 1.8,
  cex.axis = 1.8
)

grid(lwd = 2)

lines(x, qx, lty = 2, lwd = 4, col = "blue")

# 2nd order Edgeworth approximation
cumulants2 <- c(m[1], m[2]^2)
d2 <- dapx_edgeworth(x, cumulants2)
lines(x, d2, lty = 2, lwd = 4, col = "blue")

# 3rd order Edgeworth approximation
cumulants3 <- c(
  m[1],
  m[2]^2,
  m[3] * m[2]^3
)

d3 <- dapx_edgeworth(x, cumulants3)
lines(x, d3, lty = 2, lwd = 4, col = "green")

# 4th order Edgeworth approximation
cumulants4 <- c(
  m[1],
  m[2]^2,
  0.5 * m[3] * m[2]^3,
  0.2 * m[4] * m[2]^4
)

d4 <- dapx_edgeworth(x, cumulants4)
lines(x, d4, lty = 2, lwd = 4, col = "purple")

legend(
  "topleft",
  legend = c(
    "Empirical density",
    "Gaussian density",
    "3rd order Edgeworth",
    "4th order Edgeworth"
  ),
  col = c("red", "blue", "green", "purple"),
  lty = c(1, 2, 2, 2),
  lwd = 4,
  cex = 1.3
)