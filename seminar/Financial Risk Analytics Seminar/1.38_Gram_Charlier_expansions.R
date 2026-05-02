library(quantmod)

getSymbols("^DJI", from = "1990-01-03", to = Sys.Date(), src = "yahoo")

stock <- Ad(DJI)
stock_rtn <- diff(log(stock))

data <- as.vector(na.omit(stock_rtn))

m <- mean(data)
s <- sd(data)

dens <- density(data)

x <- seq(-0.25, 0.25, length.out = 1000)
y_emp <- approx(dens$x, dens$y, xout = x)$y

y_gauss <- dnorm(x, mean = m, sd = s)

# Moments
mu3 <- mean((data - m)^3)
mu4 <- mean((data - m)^4)

z <- (x - m) / s

c3 <- mu3 / (6 * s^3)
c4 <- (mu4 - 3 * s^4) / (24 * s^4)

g <- dnorm(x, mean = m, sd = s)

# Gram-Charlier approximations
d3 <- g * (1 + c3 * (z^3 - 3 * z))

d4 <- g * (
  1 +
    c3 * (z^3 - 3 * z) +
    c4 * (z^4 - 6 * z^2 + 3)
)

# Plot
dev.new(width = 16, height = 8)

plot(
  x, y_emp,
  type = "l",
  lwd = 4,
  col = "red",
  xlab = "x",
  ylab = "Density",
  main = "Empirical Density vs Gram-Charlier Approximation",
  xlim = c(-0.1, 0.1),
  ylim = c(0, 65),
  cex.lab = 1.8,
  cex.axis = 1.8
)

lines(x, y_gauss, lty = 2, lwd = 4, col = "blue")
lines(x, d3, lty = 2, lwd = 4, col = "green")
lines(x, d4, lty = 2, lwd = 4, col = "purple")

grid(lwd = 2)

legend(
  "topleft",
  legend = c(
    "Empirical density",
    "Gaussian density",
    "3rd order Gram-Charlier",
    "4th order Gram-Charlier"
  ),
  col = c("red", "blue", "green", "purple"),
  lty = c(1, 2, 2, 2),
  lwd = 4,
  cex = 1.2
)