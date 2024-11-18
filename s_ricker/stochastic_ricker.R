#=================================#
# The forecast cycle based on M.D #
#=================================#
library(zoo)
set.seed(42)
# Setup


sample_pars <- function(ns, r, k, sigma.N, error_size){
  pars <- list()
  # univariate priors: Expert opinion
  pars$r <- rnorm(ns, r, error_size*r)
  pars$k <- rnorm(ns, k, error_size*k)
  pars$sigma.N <- rnorm(ns, sigma.N, error_size*sigma.N )
  pars <- as.data.frame(pars)
  return(pars)
}

# Specify model
ricker.sim <- function(X, params){
  N_new <- X*exp(params$r*(1-X/params$k)) # + rnorm(length(params$sigma.N),0,params$sigma.N)
  return(N_new)
}

# Parameters: Priors and direct constraints
run_simulation <- function(r, k, N_init, sigma.N, error_size){
  
  params <- list()
  params$r <- rnorm(ne, r, error_size*r)
  params$k <- rnorm(ne, k, error_size*k)
  params$sigma.N <- rnorm(ne, sigma.N, error_size*sigma.N )
  params <- as.data.frame(params)
  plot(params)
  # process error: Expert opinion
  #hist(sqrt(rgamma(ne, 21, 1 * 0.05 ^ 2)))
  #params$sigma.N <- 1 / sqrt(rgamma(ne, 20, 1 * 0.05 ^ 2))
  
  X <-  rnorm(length(params$sigma.N), N_init, params$sigma.N)
  hist(X)
  
  # Forecast step: Monte Carlo Error Propagation
  # Ensemble forecast
  ensemble_forecast <- function(X,params, nt){
    output = array(0.0, c(nt, ne, 1)) 
    output[1, , ] <- X #ricker.sim(X, params)
    for(t in 2:nt){
      params <- sample_pars(ne, r, k, sigma.N, error_size)
      X <- output[t-1, , 1] 
      output[t, , ] <- ricker.sim(X, params)  ## run model, save output
    }
    output[is.nan(output)] = 0
    output[is.infinite(output)] = 0
    # output[output < 0 ] = 0
    return(output)
  }
  
  output <- ensemble_forecast(X, params, nt = horiz)
  matplot(output[,,1], type="l", col = "lightgray")
  
  return(output)
}

ne = 500 # Ensemble size. production run should be 200 - 5000, depending on what your computer can handle
horiz = 100 #days, forecast horizon during forecast

# Initial conditions
N_init <- 1
r <- 0.1
k <- 2
sigma.N <- 0.01

error_size <- 0.1
threshold <- 0.05

output <- run_simulation(r = r, k=k, N_init = N_init, sigma.N = sigma.N, error_size = error_size)

output_med <- apply(output[,,1], 1, quantile, 0.5)
output_qupper <- apply(output[,,1], 1, quantile, 0.975)
output_qlower <- apply(output[,,1], 1, quantile, 0.025)

lines(output_qupper,type="l", lty=2)
lines(output_med)
lines(output_qlower, lty=2)

ae <- function(x, y){abs(x-y)}

ens_mean <- apply(output[,,1], 1, mean)
ens_spread <- apply(output[,,1], 1, sd)

cor(ens_mean, ens_spread)

# Assessment
## model
#ci = apply(output[, ,1], 1, quantile, c(0.025, 0.5, 0.975)) ## forecast confidence interval
observations <- function(r, k, N_init, sigma.N, error_size, tsteps){
  
  params_true <- list()
  params_true$r <- r
  params_true$k <- k
  params_true$sigma.N <- sigma.N
  dyn <- matrix(nrow = tsteps, ncol=2)
  dyn[1,1] <- N_init
  dyn[1,2] <- rnorm(1, dyn[1,1], params_true$sigma.N)
  for (i in 2:tsteps){
    dyn[i,1] <- ricker.sim(dyn[i-1, 1], params_true) 
    dyn[i,2] <- rnorm(1, dyn[i,1], params_true$sigma.N)
    params_true <- sample_pars(1, r, k, sigma.N, error_size)
  }
  
  dat <- as.data.frame(dyn)
  colnames(dat) <- c("dyn.true", "dyn.proc")
  dat$sigma.N <- sqrt((dat$dyn.true - dat$dyn.proc)^2)
  
  return(dat)
}

dat_train <- observations(r = r, k=k, N_init = N_init, sigma.N = sigma.N, error_size = error_size, tsteps = horiz)

lines(dat_train$dyn.true, type = "l", col="red")
lines(dat_train$dyn.proc)

ensemble_mean_error <- mapply(ae, ens_mean, dat_train$dyn.true)
plot(error, type = "l")
cor(error, ens_spread)

plot(ens_spread, type="l")
lines(error, type="l", col="red")

# ========== # 
# Fancy Plot #
# ========== # 

light_gray <- rgb(0.8, 0.8, 0.8, alpha=0.3)
par(mar = c(5, 5, 4, 5), cex.lab = 1.5, cex.axis = 1.2) 

matplot(output[,,1], type="l", col = light_gray, 
        lty=1, 
        xlab = "Lead time [Generation]", ylab="Relative population size", 
        cex.axis=1.5, 
        font.lab=2)
lines(output_med, type="l", 
      ylim = c(0,2), 
      lwd=2,
      col="blue")
lines(output_qupper, lty=1, 
      lwd=2, col="blue")
lines(output_qlower, lty=1,
      lwd=2, col="blue")

lines(dat_train$dyn.true, type = "l",lwd=2, col = "magenta")
#lines(dat_train$dyn.proc, col= "red")
abline(h = 0.01, col="black")

legend("topleft", legend = c("Ensemble Quantiles", "Observations"),
       col = c("blue", "magenta"),
       lty = c(1, 1, 1, 1), 
       lwd = c(1, 1.5, 1.5, 1.5), 
       bty = "n",
       cex=1.3)

# =============#
# Error and CIs. 
# =============#

ensemble_error <- apply(output[,,1], 2, function(row, true_value) {
  ae(row, true_value)
}, true_value = dat_train$dyn.true)

med_ensemble_error <- apply(ensemble_error, 1, median)
qu_ensemble_error <- apply(ensemble_error, 1, quantile, 0.975)
ql_ensemble_error <- apply(ensemble_error, 1, quantile, 0.025)

#q_error_u <- mapply(ae, ens_mean, output_qupper)
#q_error_l <- mapply(ae, ens_mean, output_qlower)
# Initial plot with the primary y-axis
light_gray <- rgb(0.8, 0.8, 0.8, alpha=0.3)
par(mar = c(5, 5, 4, 5), cex.lab = 1.5, cex.axis = 1.2) 
matplot(ensemble_error, type = "l", lty =1, col = light_gray, 
        ylab = "Error", xlab="Lead time [Generation]", 
        font.lab=2)

# Add lines to the plot
lines(med_ensemble_error, col = "black", lty = 1, lwd = 1.5)
lines(ensemble_mean_error, col = "black", lty = 2, lwd = 1.5)
lines(qu_ensemble_error, col = "blue", lty = 1, lwd = 1.5)
lines(ql_ensemble_error, col = "blue", lty = 1, lwd = 1.5)
lines(2 * ens_spread, col = "red", lty = 1, lwd = 1.5)

# Add legend
legend("topleft", 
       legend = expression(Q["0.975, 0.025"]^{epsilon}, 
                           Q["0.5"]^{epsilon}, 
                           bar(epsilon), 
                           epsilon, 
                           2*s),
       col = c("blue", "black","black", "gray", "red"),
       lty = c(1, 1, 2, 1, 1), 
       lwd = c(1.5, 1.5, 1, 1.5), 
       bty = "n",
       cex = 1.5)

# Add secondary axis
axis(4, at = pretty(range(ensemble_error)), labels = pretty(range(ensemble_error)))  # Add the axis on the right
mtext("Spread", side = 4, line = 3, cex = 1.5,
      font=2)  # Add the label for the secondary axis


# =============#
# Spread-error 
# =============#



data_matrix <- cbind(error, ens_spread)

# Rolling correlation with a window size of 5
rolling_cor <- rollapplyr(
  data = data_matrix,
  width = 10,
  FUN = function(window) cor(window[, 1], window[, 2]),
  by.column = FALSE,
  align = "right"
)


par(mar = c(5, 5, 4, 5), cex.lab = 1.5, cex.axis = 1.3) 
plot(rolling_cor, type="l",lwd=2,
     xlab = "Lead time [Generation]",
     ylab= "Spread-error correlation",
     font.lab=2, ylim=c(-1,1))
abline(h=0, lty=2)






#============================#
# Quantify effect of sigma.N #
#============================#

sensitvity_horizons <- function(N_init = 1,
                              r = 0.007,
                              k = 2,
                              sigma.N = 0.01,
                              error_size_r = 0.12,
                              error_size_k = 0.12,
                              error_size_s = 0.12,
                              threshold = 0.05){

  # Parameters: Priors and direct constraints
  
  params <- list()
  params$r <- rnorm(ne, r, error_size_r*r)
  params$k <- rnorm(ne, k, error_size_k*k)
  params$sigma.N <- rnorm(ne, sigma.N, error_size_s*sigma.N )
  params <- as.data.frame(params)

  X <- N_init + rnorm(length(params$sigma.N), 0, params$sigma.N)
  
  sample_pars <- function(ns, r, k, sigma.N, 
                          error_size_r, error_size_k, error_size_s){
    pars <- list()
    # univariate priors: Expert opinion
    pars$r <- rnorm(ns, r, error_size_r*r)
    pars$k <- rnorm(ns, k, error_size_k*k)
    pars$sigma.N <- rnorm(ns, sigma.N, error_size_s*sigma.N )
    pars <- as.data.frame(pars)
    return(pars)
  }
  
  # Forecast step: Monte Carlo Error Propagation
  # Ensemble forecast
  ensemble_forecast <- function(X,params, nt){
    output = array(0.0, c(nt, ne, 1)) 
    output[1, , ] <- ricker.sim(X, params)
    for(t in 2:nt){
      params <- sample_pars(ne, params$r, params$k, params$sigma.N, 
                            error_size_r, error_size_k, error_size_s)
      X <- output[t-1, , 1] 
      output[t, , ] <- ricker.sim(X, params)  ## run model, save output
    }
    output[is.nan(output)] = 0
    output[is.infinite(output)] = 0
    # output[output < 0 ] = 0
    return(output)
  }
  
  output <- ensemble_forecast(X, params, nt = horiz)
  output_med <- apply(output[,,1], 1, quantile, 0.5)
  output_qupper <- apply(output[,,1], 1, quantile, 0.95)
  output_qlower <- apply(output[,,1], 1, quantile, 0.05)
  
  h_lower_bound <- if (any(output_qlower < threshold)) {
    which.min(output_qlower > threshold)
  } else {
    horiz
  }
  h_upper_bound <- if (any(output_qupper < dat_train$dyn.true)) {
    which.min(output_qupper > dat_train$dyn.true)
  } else {
    horiz
  }
  
  horizons <- list(h_lower_bound, h_upper_bound)
  return(horizons)
}

h_sensitivity <- data.frame(
  error_sizes = seq(0.02, 0.5, length=100),
  h_lower_bound = NA,
  h_upper_bound = NA,
  h_lower_bound_r = NA,
  h_upper_bound_r = NA,
  h_lower_bound_k = NA,
  h_upper_bound_k = NA,
  h_lower_bound_s = NA,
  h_upper_bound_s = NA)

for (i in 1:100){
  horizons <- sensitvity_horizons(error_size_r = h_sensitivity$error_sizes[i],
                                  error_size_k = h_sensitivity$error_sizes[i],
                                  error_size_s = h_sensitivity$error_sizes[i])
  h_sensitivity$h_lower_bound[i] <- horizons[[1]]
  h_sensitivity$h_upper_bound[i] <- horizons[[2]]
  print(i)
}

for (i in 1:100){
  horizons <- sensitvity_horizons(error_size_r = h_sensitivity$error_sizes[i],
                                  error_size_k = 0.12,
                                  error_size_s = 0.12)
  h_sensitivity$h_lower_bound_r[i] <- horizons[[1]]
  h_sensitivity$h_upper_bound_r[i] <- horizons[[2]]
  print(i)
}

for (i in 1:100){
  horizons <- sensitvity_horizons(error_size_r = 0.12,
                                  error_size_k = h_sensitivity$error_sizes[i],
                                  error_size_s = 0.12)
  h_sensitivity$h_lower_bound_k[i] <- horizons[[1]]
  h_sensitivity$h_upper_bound_k[i] <- horizons[[2]]
  print(i)
}

for (i in 1:100){
  horizons <- sensitvity_horizons(error_size_r = 0.12,
                                  error_size_k = 0.12,
                                  error_size_s = h_sensitivity$error_sizes[i])
  h_sensitivity$h_lower_bound_s[i] <- horizons[[1]]
  h_sensitivity$h_upper_bound_s[i] <- horizons[[2]]
  print(i)
}

par(cex.lab = 1.5, mar = c(5, 5, 4, 2)) 
plot(h_sensitivity$error_sizes, h_sensitivity$h_lower_bound, 
     type="l", col="magenta", ylim = c(0,500),
     lwd=2,
     xlab = "Prior variance factor", ylab="Horizon",
     cex.axis=1.5, 
     font.lab=2
     )
lines(h_sensitivity$error_sizes, h_sensitivity$h_upper_bound, type="l", col="purple",
      lwd=2,)
lines(h_sensitivity$error_sizes, h_sensitivity$h_lower_bound_r, type="l", col="cyan",
      lwd=2,lty=2)
lines(h_sensitivity$error_sizes, h_sensitivity$h_upper_bound_r, type="l", col="blue",
      lwd=2,lty=2)
lines(h_sensitivity$error_sizes, h_sensitivity$h_lower_bound_k, type="l", col="green",
      lwd=2,lty=3)
lines(h_sensitivity$error_sizes, h_sensitivity$h_upper_bound_k, type="l", col="darkgreen",
      lwd=2,lty=3)
legend("topright", legend =c(expression(Total[Lower]), expression(Total[Upper]),
                             expression(r[Lower]), expression(r[Upper]),
                             expression(k[Lower]), expression(k[Upper])),
       col = c("magenta", "purple", "cyan", "blue", "green", "darkgreen"),
       lty = c(1, 1, 2, 2, 3, 3), 
       lwd = c(2.5,  2.5, 2.5,  2.5, 2.5,  2.5), 
       bty = "n",
       cex=1.3)

