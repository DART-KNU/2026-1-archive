N <- 2000
t <- 0:N
dt <- 1 / N

mu <- 0.5
sigma <- 0.2
nsim <- 10

S <- matrix(0, nrow = nsim, ncol = N + 1)

# Brownian increments
Z <- matrix(
  rnorm(nsim * N, mean = 0, sd = sqrt(dt)),
  nrow = nsim,
  ncol = N
)

# Numerical solution (Euler scheme)
for (i in 1:nsim) {
  S[i, 1] <- 1.0
  
  for (j in 2:(N + 1)) {
    S[i, j] <- S[i, j - 1] * (1 + mu * dt + sigma * Z[i, j - 1])
  }
}

# Plot with title
plot(
  t * dt, S[1, ],
  type = "l",
  xlab = "Time",
  ylab = "Geometric Brownian Motion",
  main = "Numerical Solution of GBM",
  lwd = 2,
  ylim = c(min(S), max(S)),
  col = 1,
  las = 1,
  cex.axis = 1.5,
  cex.lab = 1.5,
  cex.main = 1.8,
  xaxs = "i",
  yaxs = "i"
)

for (i in 2:nsim) {
  lines(t * dt, S[i, ], lwd = 2, col = i)
}
