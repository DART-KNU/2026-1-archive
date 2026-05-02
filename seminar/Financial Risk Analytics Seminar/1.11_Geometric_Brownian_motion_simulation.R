N <- 1000
t <- 0:N
dt <- 1 / N

sigma <- 0.2
mu <- 0.5

# Brownian increments
Z <- rnorm(N, mean = 0, sd = sqrt(dt))

# Brownian motion path
B <- c(0, cumsum(Z))

# Deterministic growth path
deterministic_path <- exp(mu * t * dt)

# Geometric Brownian motion path
gbm_path <- exp(
  sigma * B + 
    mu * t * dt - 
    0.5 * sigma^2 * t * dt
)

# Plot deterministic path
plot(
  t * dt,
  deterministic_path,
  xlab = "Time",
  ylab = "Geometric Brownian Motion",
  type = "l",
  ylim = c(0.75, 2),
  col = "black",
  lwd = 3,
  main = "Geometric Brownian Motion Simulation"
)

# Add GBM path
lines(
  t * dt,
  gbm_path,
  col = "blue",
  lwd = 1.2,
  xaxs = "i",
  yaxs = "i"
)

legend(
  "topleft",
  legend = c("Deterministic Growth", "GBM Path"),
  col = c("black", "blue"),
  lwd = c(3, 1.2),
  bty = "n"
)