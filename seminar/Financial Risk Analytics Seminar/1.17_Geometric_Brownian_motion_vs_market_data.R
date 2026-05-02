install.packages("quantmod")
install.packages("Sim.DiffProc")

library(quantmod)
library(Sim.DiffProc)

getSymbols("0005.HK", from = "2016-02-15", to = "2017-05-11", src = "yahoo")

MP <- Ad(`0005.HK`)

fx <- expression(theta[1] * x)
gx <- expression(theta[2] * x)

MP_ts <- as.ts(as.numeric(MP))

fit <- fitsde(
  data = MP_ts,
  drift = fx,
  diffusion = gx,
  start = list(theta1 = 0.01, theta2 = 0.01),
  pmle = "euler"
)

fit