library(quantmod)

# Data
getSymbols("^GSPC", from = "1950-01-01", to = "2022-12-31", src = "yahoo")

stock_price <- Cl(GSPC)

s <- 0
y <- 0
j <- 0
ct <- 0

N <- 240
nsim <- 72
X <- matrix(0, nrow = nsim, ncol = N)

# Extract yearly paths
for (i in 1:nrow(stock_price)) {
  current_date <- as.character(index(stock_price[i]))
  
  if (s == 0 && grepl("-01-0", current_date)) {
    if (ct == 0 || X[y, N] > 0) {
      y <- y + 1
      j <- 1
      s <- 1
      ct <- ct + 1
    }
  }
  
  if (j <= N && y > 0 && y <= nsim) {
    X[y, j] <- as.numeric(stock_price[i])
  }
  
  if (grepl("-02-0", current_date)) {
    s <- 0
  }
  
  j <- j + 1
}

# Annual return ratio
stock <- X[, N] / X[, 1]

# Empirical density
stk.dens <- density(stock, na.rm = TRUE)

# Lognormal density
x <- seq(min(stk.dens$x), 1.2 * max(stk.dens$x), length.out = 100)

qx <- dlnorm(
  x,
  meanlog = mean(log(stock), na.rm = TRUE),
  sdlog = sd(log(stock), na.rm = TRUE)
)

# Plot
dev.new(width = 10, height = 5)

plot(
  x, qx,
  type = "l",
  lty = 2,
  lwd = 3,
  col = "blue",
  xlab = "Prices",
  ylab = "Density",
  ylim = c(0, max(stk.dens$y, qx)),
  main = "Empirical vs Lognormal Density",
  panel.first = abline(h = 0, col = "grey", lwd = 0.2)
)

lines(
  stk.dens,
  lwd = 3,
  col = "red"
)

legend(
  "topright",
  legend = c("Emp. density", "Lognormal density"),
  col = c("red", "blue"),
  lty = c(1, 2),
  lwd = 3,
  cex = 1.2
)