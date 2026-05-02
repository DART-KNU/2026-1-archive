beta <- 0.5
nsim <- 100
N <- 1000

t <- 0:N
dt <- 1 / N

# Generate random walk increments
dX <- (dt)^beta * (rbinom(nsim * N, 1, 0.5) - 0.5) * 2
dX <- matrix(dX, nrow = nsim, ncol = N)

# Construct cumulative paths starting from 0
X <- cbind(rep(0, nsim), t(apply(dX, 1, cumsum)))

# Histogram of terminal values
H <- hist(X[, N + 1], plot = FALSE)

# Two-panel layout
layout(matrix(c(1, 2), nrow = 1, byrow = TRUE))

# Left plot: sample paths
par(mar = c(4, 4, 3, 1))

plot(
  t * dt, X[1, ],
  type = "l",
  ylim = c(-2, 2),
  col = 1,
  xaxs = "i",
  las = 1,
  xlab = "time",
  ylab = "X(t)",
  main = "Simulated Brownian Motion Paths"
)

for (i in 1:nsim) {
  lines(t * dt, X[i, ], col = i)
}

# Theoretical scale: ±sqrt(t)
lines(t * dt, sqrt(t * dt), col = "red", lwd = 3)
lines(t * dt, -sqrt(t * dt), col = "red", lwd = 3)

# Zero line
abline(h = 0, lwd = 2)

# Terminal points
for (i in 1:nsim) {
  points(1, X[i, N + 1], pch = 1, lwd = 3, col = i)
}

# Right plot: terminal distribution
x <- seq(-2, 2, length.out = 100)
px <- dnorm(x)

par(mar = c(4, 2, 3, 2))

plot(
  NULL,
  xlab = "Density",
  ylab = "",
  xlim = c(0, max(px, H$density)),
  ylim = c(-2, 2
  axes = FALSE,
  main = "Terminal Distribution"
)

axis(1)
axis(2, las = 1)

rect(
  xleft = 0,
  ybottom = H$breaks[-length(H$breaks)],
  xright = H$density,
  ytop = H$breaks[-1],
  col = rainbow(length(H$density), start = 0.08, end = 0.6),
  border = "white"
)

lines(px, x, lwd = 2)

box()