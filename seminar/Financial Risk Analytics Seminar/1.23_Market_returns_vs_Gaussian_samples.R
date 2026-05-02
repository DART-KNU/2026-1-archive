library(quantmod)

dev.new(width = 16, height = 8)

# Fetch Dow Jones data
getSymbols("^DJI", from = "1990-01-03", to = Sys.Date(), src = "yahoo")

# Adjusted close price
stock <- Ad(DJI)

# Log returns
stk.rtn <- diff(log(stock))

rt <- as.vector(stk.rtn)

m <- mean(rt, na.rm = TRUE)
s <- sd(rt, na.rm = TRUE)

ts <- index(stk.rtn)
n <- length(rt)

# Normally simulated returns with same mean and sd
y <- rnorm(n, mean = m, sd = s)

# Plot actual returns
plot(
  ts, rt,
  pch = 19,
  xaxs = "i",
  yaxs = "i",
  cex = 0.03,
  col = "blue",
  ylab = "",
  xlab = "",
  main = "Dow Jones Log Returns vs Normal Simulation",
  las = 1,
  cex.lab = 1.8,
  cex.axis = 1.8,
  lwd = 3
)

segments(
  x0 = ts,
  x1 = ts,
  y0 = 0,
  y1 = rt,
  col = "blue"
)

# Add simulated normal returns
points(ts, y, pch = 19, cex = 0.3, col = "red")

# Mean and ±3 sigma lines
abline(h = m + 3 * s, lwd = 1)
abline(h = m, lwd = 1)
abline(h = m - 3 * s, lwd = 1)

# Tail probabilities
actual_tail_prob <- sum(abs(rt - m) > 3 * s, na.rm = TRUE) / sum(!is.na(rt))
sim_tail_prob <- sum(abs(y - m) > 3 * s) / length(y)
normal_tail_prob <- 2 * (1 - pnorm(3))

actual_tail_prob
sim_tail_prob
normal_tail_prob