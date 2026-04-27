N <- 1000
t <- 0:N
dt <- 1 / N
ns <- 100

sigma <- 0.2
r <- 0.5

# Bernoulli-based one-step returns
a <- (1 + r * dt) * (1 - sigma * sqrt(dt)) - 1
b <- (1 + r * dt) * (1 + sigma * sqrt(dt)) - 1

# Simulated returns
X <- matrix(
  a + (b - a) * rbinom(ns * N, 1, 0.5),
  nrow = ns,
  ncol = N
)

# Convert returns to price paths
X <- cbind(1, t(apply(1 + X, 1, cumprod)))

# Histogram of terminal values
H <- hist(X[, N + 1], plot = FALSE)

# Two-panel layout
dev.new(width = 16, height = 7)
layout(matrix(c(1, 2), nrow = 1, byrow = TRUE))

par(mar = c(2, 2, 2, 0), oma = c(2, 2, 2, 2))

# Left plot: simulated GBM paths
plot(
  t * dt,
  X[1, ],
  xlab = "Time",
  ylab = "",
  type = "l",
  ylim = c(0.8, 3),
  col = 0,
  xaxs = "i",
  las = 1,
  cex.axis = 1.6,
  main = "GBM Simulation"
)

for (i in 1:ns) {
  lines(t * dt, X[i, ], col = i)
}

# Expected growth path
lines(
  t * dt,
  (1 + r * dt)^t,
  type = "l",
  lty = 1,
  lwd = 3,
  col = "black"
)

# Terminal points
for (i in 1:ns) {
  points(1, X[i, N + 1], pch = 1, lwd = 3, col = i)
}

# Right plot: histogram + lognormal density
x <- seq(0.01, 3, length.out = 100)

px <- exp(
  - (log(x) - (r - sigma^2 / 2))^2 / (2 * sigma^2)
) / (x * sigma * sqrt(2 * pi))

pl <- rainbow(length(H$density), start = 0.08, end = 0.6)

par(mar = c(2, 2, 2, 2))

plot(
  NULL,
  xlab = "",
  ylab = "",
  xlim = c(0, max(px, H$density)),
  ylim = c(0.8, 3),
  axes = FALSE,
  las = 1,
  main = "Terminal Lognormal Distribution"
)

rect(
  xleft = 0,
  ybottom = H$breaks[-length(H$breaks)],
  xright = H$density,
  ytop = H$breaks[-1],
  col = pl,
  border = "white"
)

lines(px, x, type = "l", lty = 1, lwd = 3, col = "black")