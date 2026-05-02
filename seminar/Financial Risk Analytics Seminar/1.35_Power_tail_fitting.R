install.packages("pracma")
library(pracma)

# Grid
x <- seq(-0.25, 0.25, length.out = 1000)

# Empirical density
stk.dens <- density(
  stk.rtn,
  na.rm = TRUE,
  from = -0.1,
  to = 0.1,
  n = 1000
)

# Rational function fitting
a <- rationalfit(
  stk.dens$x,
  stk.dens$y,
  d1 = 2,
  d2 = 2
)

# Fitted density
power_density <- (
  a$p1[3] + a$p1[2] * x + a$p1[1] * x^2
) / (
  a$p2[3] + a$p2[2] * x + a$p2[1] * x^2
)

# Plot
dev.new(width = 16, height = 8)

plot(
  stk.dens$x,
  stk.dens$y,
  type = "l",
  lwd = 4,
  col = "red",
  xlab = "",
  ylab = "",
  main = "Empirical Density vs Power Density",
  xlim = c(-0.1, 0.1),
  ylim = c(0, 65),
  xaxs = "i",
  yaxs = "i",
  las = 1,
  cex.lab = 1.8,
  cex.axis = 1.8
)

lines(
  x,
  power_density,
  type = "l",
  lty = 2,
  col = "blue",
  lwd = 4
)

grid(lwd = 2)

legend(
  "topleft",
  legend = c("Empirical density", "Power density"),
  col = c("red", "blue"),
  lty = c(1, 2),
  lwd = 4,
  cex = 1.5
)