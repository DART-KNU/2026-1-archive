# Empirical CDF
stock.ecdf <- ecdf(as.vector(stk.rtn))

# Grid for theoretical CDF
x <- seq(-0.15, 0.15, length.out = 200)

# Gaussian CDF
px <- pnorm((x - m) / s)

# Plot
dev.new(width = 16, height = 8)

plot(
  stock.ecdf,
  xlab = "",
  ylab = "",
  col = "red",
  ylim = c(-0.002, 1.002),
  main = "Empirical vs Gaussian CDF",
  xaxs = "i",
  yaxs = "i",
  las = 1,
  cex.lab = 1.8,
  cex.axis = 1.8,
  lwd = 4
)

# Add theoretical normal CDF
lines(
  x, px,
  col = "blue",
  lty = 2,
  lwd = 4
)

grid(lwd = 3)

legend(
  "topleft",
  legend = c("Empirical CDF", "Gaussian CDF"),
  col = c("red", "blue"),
  lty = c(1, 2),
  lwd = 4,
  cex = 1.5
)