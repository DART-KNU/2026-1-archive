# Install and load package
install.packages("quantmod")
library(quantmod)

# Fetch stock data from Yahoo Finance
getSymbols("AAPL", src = "yahoo")
getSymbols("GOOG", src = "yahoo")

# Fetch Dow Jones Industrial Average data
getSymbols("^DJI", from = "2007-01-03", to = Sys.Date(), src = "yahoo")

# Use adjusted closing price
stock <- Ad(DJI)

# Plot stock price
dev.new(width = 12, height = 8)
chartSeries(
  stock,
  up.col = "blue",
  theme = "white",
  name = "Dow Jones Industrial Average"
)

# Log returns
stock.logrtn <- diff(log(stock))

# Simple returns
stock.rtn <- (stock - lag(stock)) / lag(stock)

# Plot simple returns
dev.new(width = 12, height = 8)
chartSeries(
  stock.rtn,
  up.col = "blue",
  theme = "white",
  name = "DJI Simple Returns"
)

# Number of valid observations
n <- sum(!is.na(stock.rtn))

n
