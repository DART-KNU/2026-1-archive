dev.new(width = 16, height = 8)

qqnorm(
  stk.rtn,
  col = "blue",
  pch = 16,
  xaxs = "i",
  yaxs = "i",
  las = 1,
  cex.lab = 1.4,
  cex.axis = 1,
  main = "QQ Plot of Log Returns"
)

qqline(stk.rtn, col = "red", lwd = 2)

grid(lwd = 2)