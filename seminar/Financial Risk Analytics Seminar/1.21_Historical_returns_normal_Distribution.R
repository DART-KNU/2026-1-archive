library(quantmod)

getSymbols("^GSPC", from = "1950-01-01", to = "2022-12-31", src = "yahoo")

stock <- Cl(GSPC)

s <- 0
y <- 0
j <- 0
ct <- 0

N <- 240
nsim <- 72
X <- matrix(0, nrow = nsim, ncol = N)

# Extract yearly paths starting in early January
for (i in 1:nrow(stock)) {
  
  current_date <- as.character(index(stock[i]))
  
  if (s == 0 && grepl("-01-0", current_date)) {
    if (ct == 0 || X[y, N] > 0) {
      y <- y + 1
      j <- 1
      s <- 1
      ct <- ct + 1
    }
  }
  
  if (j <= N && y <= nsim && y > 0) {
    X[y, j] <- as.numeric(stock[i])
  }
  
  if (grepl("-02-0", current_date)) {
    s <- 0
  }
  
  j <- j + 1
}

t <- 0:(N - 1)
dt <- 1 / N

# Drift and volatility from yearly returns
m <- mean(X[, N] / X[, 1] - 1, na.rm = TRUE)
sg <- sd(X[, N] / X[, 1] - 1, na.rm = TRUE)

dev.new(width = 16, height = 7)

layout(matrix(c(1, 2), nrow = 1))
par(mar = c(2, 2, 2, 0), oma = c(2, 2, 2, 2))

# Left plot: normalized yearly paths
plot(
  t * dt,
  X[1, ] / X[1, 1] - 1 - m * t * dt,
  xlab = "",
  ylab = "",
  type = "l",
  ylim = c(-0.5, 0.5),
  col = 0,
  xaxs = "i",
  las = 1,
  cex.axis = 1.6,
  main = "Normalized S&P 500 Yearly Paths"
)

for (i in 1:nsim) {
  lines(
    t * dt,
    X[i, ] / X[i, 1] - 1 - m * t * dt,
    type = "l",
    col = i
  )
}

# ± volatility sqrt(t)
lines(t * dt, sg * sqrt(t * dt), col = "red", lwd = 3)
lines(t * dt, -sg * sqrt(t * dt), col = "red", lwd = 3)

# Zero line
lines(t * dt, 0 * t, col = "black", lwd = 2)

# Terminal points
for (i in 1:nsim) {
  points(
    0.999,
    X[i, N] / X[i, 1] - 1 - m * N * dt,
    pch = 1,
    lwd = 5,
    col = i
  )
}

# Right plot: terminal distribution
x <- seq(-0.5, 0.5, length.out = 100)
px <- dnorm(x, mean = 0, sd = sg)

H <- hist(
  X[, N] / X[, 1] - 1 - m * N * dt,
  plot = FALSE
)

par(mar = c(2, 2, 2, 2))

plot(
  NULL,
  xlab = "",
  ylab = "",
  xlim = c(0, max(px, H$density)),
  ylim = c(-0.5, 0.5),
  axes = FALSE,
  main = "Terminal Distribution"
)

rect(
  xleft = 0,
  ybottom = H$breaks[-length(H$breaks)],
  xright = H$density,
  ytop = H$breaks[-1],
  col = rainbow(length(H$density), start = 0.08, end = 0.6),
  border = "white"
)

lines(px, x, col = "black", lwd = 2)