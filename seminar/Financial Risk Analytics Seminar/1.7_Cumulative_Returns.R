# returns 계산
stock.rtn <- (stock - lag(stock)) / lag(stock)

# NA 제거
valid_idx <- !is.na(stock.rtn)

stock <- stock[valid_idx]
stock.rtn <- stock.rtn[valid_idx]

# 시간 index
times <- index(stock)

# plot
dev.new(width = 16, height = 7)
par(mfrow = c(1, 2))

# returns plot
plot(
  times, stock.rtn,
  pch = 19,
  cex = 0.03,
  col = "blue",
  main = "Asset Returns"
)

segments(
  times, 0,
  times, as.numeric(stock.rtn),
  col = "blue"
)

# cumulative price
plot(
  times,
  100 * cumprod(1 + as.numeric(stock.rtn)),
  type = "l",
  col = "black",
  main = "Asset Prices"
)