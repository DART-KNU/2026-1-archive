# Sample size
n <- length(stk.rtn)

# Simulated normal sample
y <- rnorm(n, mean = m, sd = s)

# KS test on simulated normal data
ks_sim <- ks.test(y, "pnorm", mean = m, sd = s)

# KS test on actual returns (NA 제거 필수)
ks_real <- ks.test(na.omit(stk.rtn), "pnorm", mean = m, sd = s)

ks_sim
ks_real