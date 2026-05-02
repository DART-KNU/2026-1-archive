N <- 2000
t <- 0:N
dt <- 1 / N

mu <- 0.5
sigma <- 0.2
nsim <- 10

par(oma = c(0, 1, 0, 0))

# Brownian increments
X <- matrix(
  rnorm(nsim * N, mean = 0, sd = sqrt(dt)),
  nrow = nsim,
  ncol = N
)

# Brownian motion paths
X <- cbind(0, t(apply(X, 1, cumsum)))

# GBM transformation
for (i in 1:nsim) {
  X[i, ] <- exp(mu * t * dt + sigma * X[i, ] - 0.5 * sigma^2 * t * dt)
}

# Plot expectation
plot(
  t * dt,
  exp(mu * t * dt),
  xlab = "Time",
  ylab = "Geometric Brownian Motion",
  lwd = 3,
  ylim = c(min(X), max(X)),
  type = "l",
  col = "black",
  las = 1,
  cex.axis = 1.5,
  cex.lab = 1.6,
  xaxs = "i",
  yaxs = "i",
  main = "GBM Simulation (Multiple Paths)"
)

# Add simulated paths
for (i in 1:nsim) {
  lines(
    t * dt,
    X[i, ],
    lwd = 1.2,
    col = 2 + (i %% 7)
  )
}