dev.new(width = 16, height = 8)

# Grid for Gaussian density
x <- seq(-0.25, 0.25, length.out = 1000)
qx <- dnorm(x, mean = m, sd = s)

# Empirical density (NA 제거)
stk.dens <- density(stk.rtn, na.rm = TRUE)

# Plot empirical density
plot(
  stk.dens,
  xlab = "x",
  ylab = "",
  col = "red",
  lwd = 4,
  main = "Empirical vs Gaussian Density",
  xlim = c(-0.1, 0.1),
  ylim = c(0, 65),
  xaxs = "i",
  yaxs = "i",
  las = 1,
  cex.lab = 1.8,
  cex.axis = 1.8
)

# Add Gaussian density
lines(
  x, qx,
  col = "blue",
  lty = 2,
  lwd = 4
)

grid(lwd = 2)

legend(
  "topleft",
  legend = c("Empirical density", "Gaussian density"),
  col = c("red", "blue"),
  lty = c(1, 2),
  lwd = 4,
  cex = 1.2
)