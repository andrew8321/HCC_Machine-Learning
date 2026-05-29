####################### Internal LT #####################
library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot) # For plot_grid

# 1. Load data
file_path <- "C:/Users/andre/Desktop/internal_validation_with_survival_columns.xlsx"
df <- read_excel(file_path)

df <- df %>%
  filter(treatment == "LT")

# 2. Define survival probabilities
df <- df %>%
  mutate(
    old_prob = surv_surgery, 
    new_prob = new_survival_probability
  )

# 3. Fit Cox model
cox_model <- coxph(Surv(survival_time, event) ~ old_prob, data = df)
basehaz_df <- basehaz(cox_model, centered = FALSE)
beta <- coef(cox_model)["old_prob"]
df$lp_new <- beta * df$new_prob

# 4. Predict survival time
get_pred_survtime <- function(lp_val) {
  surv_curve <- exp(-basehaz_df$hazard * exp(lp_val))
  idx <- which(surv_curve <= 0.5)[1]
  if (!is.na(idx)) basehaz_df$time[idx] else max(basehaz_df$time)
}
df$predicted_survival_time <- sapply(df$lp_new, get_pred_survtime)

# 5. Create combined data for KM
combined_data <- data.frame(
  time = c(df$survival_time, pmin(df$predicted_survival_time, 120)),
  event = c(df$event, ifelse(df$predicted_survival_time > 120, 0, 1)), # Event = 0 if predicted beyond 120 (censored)
  group = c(rep("Observed", nrow(df)), rep("ML-guided", nrow(df)))
)
combined_data$group <- factor(combined_data$group, levels = c('Observed',"ML-guided"))

# 6. Fit KM and Cox
km_fit <- survfit(Surv(time, event) ~ group, data = combined_data)
cox_fit <- coxph(Surv(time, event) ~ group, data = combined_data)
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients[,"exp(coef)"]
ci_lower <- summary_cox$conf.int[,"lower .95"]
ci_upper <- summary_cox$conf.int[,"upper .95"]
p_value <- signif(summary_cox$coefficients[,"Pr(>|z|)"], 3)

# 7. Compute mOS
mOS_vals <- surv_median(km_fit)$median
mOS_labels <- ifelse(
  is.na(mOS_vals) | mOS_vals > 120,
  "mOS = u.d.",
  sprintf("mOS = %.1f", mOS_vals)
)

legend_labels <- c(
  sprintf("Real-world treatment %s", mOS_labels[1]),
  sprintf("ML-guided treatment %s", mOS_labels[2])
)

# 8. Custom annotation text
p_text <- if (p_value < 0.001) expression(italic(P) < ".001") else bquote(italic(P) == .(p_value))
hr_text <- sprintf("HR = %.2f (95%% CI: %.2f–%.2f)", hr, ci_lower, ci_upper)

# 9. Plot
p_combined <- ggsurvplot(
  km_fit,
  data = combined_data,
  conf.int = FALSE,
  pval = FALSE,
  xlab = "Months elapsed",
  ylab = "Probability of survival",
  legend.title = NULL,
  legend.labs = legend_labels,
  ggtheme = theme_classic(),
  xlim = c(0, 120),
  risk.table = TRUE,
  risk.table.y.text = FALSE,
  risk.table.height = 0.25,
  break.time.by = 30,
  palette = c("blue", "red"),
  censor = FALSE,
  surv.scale = "default",
  size = 1.2,
  risk.table.fontsize = 6  # ✅ 추가됨
)

# Extract data for the ML-guided curve from km_fit
ml_guided_data <- with(summary(km_fit, times = seq(0, 120, by = 0.1)), {
  data.frame(
    time = time[strata == "group=ML-guided"],
    surv = surv[strata == "group=ML-guided"]
  )
})

if(nrow(ml_guided_data) > 0){
  last_ml_drop_time <- max(ml_guided_data$time)
  last_ml_surv_prob <- ml_guided_data$surv[ml_guided_data$time == last_ml_drop_time]
} else {
  last_ml_drop_time <- 0
  last_ml_surv_prob <- 1
}

# Ensure the curve extends horizontally after its last observed event
extended_ml_guided_data <- rbind(
  ml_guided_data,
  data.frame(time = c(last_ml_drop_time, 120), surv = c(last_ml_surv_prob, last_ml_surv_prob))
)
# Remove duplicate points at last_ml_drop_time if they exist due to rbind
extended_ml_guided_data <- extended_ml_guided_data %>% distinct(time, .keep_all = TRUE) %>% arrange(time)

# 10. Styling & annotations
n_total <- nrow(df)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("LT\n(n=%d)", n_total)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) + # Ensure main plot also has these breaks
  scale_y_continuous(
    expand = c(0, 0), limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = seq(0, 100, by = 25)
  ) +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 30, face = "bold"),
    legend.text = element_text(size = 20),
    legend.title = element_blank(),
    legend.key.height = unit(1, "cm"),
    plot.title = element_text(size = 30, face = "bold", hjust = 0.5),
    legend.position = c(1, 0.5),
    legend.justification = c(1, 1),
    plot.margin = unit(c(1,1,1,1), "cm")
  ) +
  guides(color = guide_legend(ncol = 1)) +
  annotate("text", x = 10, y = 0.2, label = p_text, size = 8, hjust = 0) +
  annotate("text", x = 10, y = 0.14, label = hr_text, size = 8, hjust = 0) +
  # Add the extended ML-guided line manually, ensuring it overlays the original KM line
  geom_line(data = extended_ml_guided_data, aes(x = time, y = surv), color = "red", size = 1.2, linetype = "solid")


# The risk table also needs theme adjustments to match the main plot's x-axis text size
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15), # Increase x-axis text size for the risk table
    axis.title.x = element_blank(), # Remove x-axis label from risk table if desired
    axis.ticks.x = element_blank(), # Remove x-axis ticks from risk table if desired
    axis.title.y = element_blank(),  # strata 제목 제거
    plot.title = element_text(size = 20, face = "bold"), # Adjust title size for risk table if present
    plot.margin = unit(c(0, 1,1,1), "cm") # Adjust margin to align with main plot
  )


# 11. Combine plot and risk table
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v", # Align vertically
  rel_heights = c(3, 1) # Adjust these values as needed
)

# 12. Render and Save as high-res PNG
print(combined_final_plot)
ggsave("Figure 3B.eps", plot = combined_final_plot, width = 10, height = 12, dpi = 300)
###################### Internal SR ##################### 
library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot) # For plot_grid

# 1. Load data
file_path <- "C:/Users/andre/Desktop/internal_validation_with_survival_columns.xlsx"
df <- read_excel(file_path)

df <- df %>%
  filter(treatment == "surgery")

# 2. Define survival probabilities
df <- df %>%
  mutate(
    old_prob = surv_surgery, # Corrected to surv_surgery since treatment is "surgery"
    new_prob = new_survival_probability
  )

# 3. Fit Cox model
cox_model <- coxph(Surv(survival_time, event) ~ old_prob, data = df)
basehaz_df <- basehaz(cox_model, centered = FALSE)
beta <- coef(cox_model)["old_prob"]
df$lp_new <- beta * df$new_prob

# 4. Predict survival time
get_pred_survtime <- function(lp_val) {
  surv_curve <- exp(-basehaz_df$hazard * exp(lp_val))
  idx <- which(surv_curve <= 0.5)[1]
  if (!is.na(idx)) basehaz_df$time[idx] else max(basehaz_df$time)
}
df$predicted_survival_time <- sapply(df$lp_new, get_pred_survtime)

# 5. Create combined data for KM

combined_data <- data.frame(
  time = c(df$survival_time, pmin(df$predicted_survival_time, 120)),
  event = c(df$event, ifelse(df$predicted_survival_time > 120, 0, 1)), 
  group = c(rep("Observed", nrow(df)), rep("ML-guided", nrow(df)))
)
combined_data$group <- factor(combined_data$group, levels = c('Observed',"ML-guided"))

# 6. Fit KM and Cox
km_fit <- survfit(Surv(time, event) ~ group, data = combined_data)
cox_fit <- coxph(Surv(time, event) ~ group, data = combined_data)
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients[,"exp(coef)"]
ci_lower <- summary_cox$conf.int[,"lower .95"]
ci_upper <- summary_cox$conf.int[,"upper .95"]
p_value <- signif(summary_cox$coefficients[,"Pr(>|z|)"], 3)

# 7. Compute mOS
mOS_vals <- surv_median(km_fit)$median
mOS_labels <- ifelse(
  is.na(mOS_vals) | mOS_vals > 120,
  "mOS = u.d.",
  sprintf("mOS = %.1f", mOS_vals)
)

legend_labels <- c(
  sprintf("Real-world treatment %s", mOS_labels[1]),
  sprintf("ML-guided treatment %s", mOS_labels[2])
)

# 8. Custom annotation text
p_text <- if (p_value < 0.001) expression(italic(P) < ".001") else bquote(italic(P) == .(p_value))
hr_text <- sprintf("HR = %.2f (95%% CI: %.2f–%.2f)", hr, ci_lower, ci_upper)

# 9. Plot
p_combined <- ggsurvplot(
  km_fit,
  data = combined_data,
  conf.int = FALSE,
  pval = FALSE,
  xlab = "Months elapsed",
  ylab = "Probability of survival",
  legend.title = NULL,
  legend.labs = legend_labels,
  ggtheme = theme_classic(),
  xlim = c(0, 120),
  risk.table = TRUE,
  risk.table.y.text = FALSE,
  risk.table.height = 0.25,
  break.time.by = 30,
  palette = c("blue", "red"),
  censor = FALSE,
  surv.scale = "default",
  size = 1.2,
  risk.table.fontsize = 6  # ✅ 추가됨
)


ml_guided_data <- with(summary(km_fit, times = seq(0, 120, by = 0.1)), {
  data.frame(
    time = time[strata == "group=ML-guided"],
    surv = surv[strata == "group=ML-guided"]
  )
})


if(nrow(ml_guided_data) > 0){
  last_ml_drop_time <- max(ml_guided_data$time)
  last_ml_surv_prob <- ml_guided_data$surv[ml_guided_data$time == last_ml_drop_time]
} else {
  # If no data points for ML-guided (e.g., all censored at time 0), assume survival 1 at time 0
  last_ml_drop_time <- 0
  last_ml_surv_prob <- 1
}

# Ensure the curve extends horizontally after its last observed event
extended_ml_guided_data <- rbind(
  ml_guided_data,
  data.frame(time = c(last_ml_drop_time, 120), surv = c(last_ml_surv_prob, last_ml_surv_prob))
)
# Remove duplicate points at last_ml_drop_time if they exist due to rbind
extended_ml_guided_data <- extended_ml_guided_data %>% distinct(time, .keep_all = TRUE) %>% arrange(time)


# 10. Styling & annotations
n_total <- nrow(df)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("SR\n(n=3,619)")) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) + # Ensure main plot also has these breaks
  scale_y_continuous(
    expand = c(0, 0), limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = seq(0, 100, by = 25)
  ) +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 30, face = "bold"),
    legend.text = element_text(size = 20),
    legend.title = element_blank(),
    legend.key.height = unit(1, "cm"),
    plot.title = element_text(size = 30, face = "bold", hjust = 0.5),
    legend.position = c(1, 0.5),
    legend.justification = c(1, 1),
    plot.margin = unit(c(1,1,1,1), "cm")
  ) +
  guides(color = guide_legend(ncol = 1)) +
  annotate("text", x = 10, y = 0.2, label = p_text, size = 8, hjust = 0) +
  annotate("text", x = 10, y = 0.14, label = hr_text, size = 8, hjust = 0) +
  # Add the extended ML-guided line manually, ensuring it overlays the original KM line
  geom_line(data = extended_ml_guided_data, aes(x = time, y = surv), color = "red", size = 1.2, linetype = "solid")


# The risk table also needs theme adjustments to match the main plot's x-axis text size
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15), # Increase x-axis text size for the risk table
    axis.title.x = element_blank(), # Remove x-axis label from risk table if desired
    axis.title.y = element_blank(),  # strata 제목 제거
    axis.ticks.x = element_blank(), # Remove x-axis ticks from risk table if desired
    plot.title = element_text(size = 20, face = "bold"), # Adjust title size for risk table if present
    plot.margin = unit(c(0, 1,1,1), "cm") # Adjust margin to align with main plot
  )


# 11. Combine plot and risk table
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v", # Align vertically
  rel_heights = c(3, 1) # Adjust these values as needed
)

# 12. Render and Save as high-res PNG
print(combined_final_plot)
ggsave("Figure 3C.eps", plot = combined_final_plot, width = 10, height = 12, dpi = 300)

###################### External_LT###############=
library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot) # For plot_grid

# 1. Load data
file_path <- "C:/Users/andre/Desktop/external_validation_with_survival_columns.xlsx"
df <- read_excel(file_path)

df <- df %>%
  filter(treatment == "LT")

# 2. Define survival probabilities
df <- df %>%
  mutate(
    old_prob = surv_surgery, # Corrected to surv_surgery since treatment is "surgery"
    new_prob = new_survival_probability
  )

# 3. Fit Cox model
cox_model <- coxph(Surv(survival_time, event) ~ old_prob, data = df)
basehaz_df <- basehaz(cox_model, centered = FALSE)
beta <- coef(cox_model)["old_prob"]
df$lp_new <- beta * df$new_prob

# 4. Predict survival time
get_pred_survtime <- function(lp_val) {
  surv_curve <- exp(-basehaz_df$hazard * exp(lp_val))
  idx <- which(surv_curve <= 0.5)[1]
  if (!is.na(idx)) basehaz_df$time[idx] else max(basehaz_df$time)
}
df$predicted_survival_time <- sapply(df$lp_new, get_pred_survtime)

# 5. Create combined data for KM
combined_data <- data.frame(
  time = c(df$survival_time, pmin(df$predicted_survival_time, 120)),
  event = c(df$event, ifelse(df$predicted_survival_time > 120, 0, 1)), 
  group = c(rep("Observed", nrow(df)), rep("ML-guided", nrow(df)))
)
combined_data$group <- factor(combined_data$group, levels = c('Observed',"ML-guided"))

# 6. Fit KM and Cox
km_fit <- survfit(Surv(time, event) ~ group, data = combined_data)
cox_fit <- coxph(Surv(time, event) ~ group, data = combined_data)
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients[,"exp(coef)"]
ci_lower <- summary_cox$conf.int[,"lower .95"]
ci_upper <- summary_cox$conf.int[,"upper .95"]
p_value <- signif(summary_cox$coefficients[,"Pr(>|z|)"], 3)

# 7. Compute mOS
mOS_vals <- surv_median(km_fit)$median
mOS_labels <- ifelse(
  is.na(mOS_vals) | mOS_vals > 120,
  "mOS = u.d.",
  sprintf("mOS = %.1f", mOS_vals)
)

legend_labels <- c(
  sprintf("Real-world treatment %s", mOS_labels[1]),
  sprintf("ML-guided treatment %s", mOS_labels[2])
)

# 8. Custom annotation text
p_text <- if (p_value < 0.001) expression(italic(P) < ".001") else bquote(italic(P) == .(p_value))
hr_text <- sprintf("HR = %.2f (95%% CI: %.2f–%.2f)", hr, ci_lower, ci_upper)

p_combined <- ggsurvplot(
  km_fit,
  data = combined_data,
  conf.int = FALSE,
  pval = FALSE,
  xlab = "Months elapsed",
  ylab = "Probability of survival",
  legend.title = NULL,
  legend.labs = legend_labels,
  ggtheme = theme_classic(),
  xlim = c(0, 120),
  risk.table = TRUE,
  risk.table.y.text = FALSE,
  risk.table.height = 0.25,
  break.time.by = 30,
  palette = c("blue", "red"),
  censor = FALSE,
  surv.scale = "default",
  size = 1.2,
  risk.table.fontsize = 6  # ✅ 추가됨
)

# Extract data for the ML-guided curve from km_fit
ml_guided_data <- with(summary(km_fit, times = seq(0, 120, by = 0.1)), {
  data.frame(
    time = time[strata == "group=ML-guided"],
    surv = surv[strata == "group=ML-guided"]
  )
})

if(nrow(ml_guided_data) > 0){
  last_ml_drop_time <- max(ml_guided_data$time)
  last_ml_surv_prob <- ml_guided_data$surv[ml_guided_data$time == last_ml_drop_time]
} else {
  # If no data points for ML-guided (e.g., all censored at time 0), assume survival 1 at time 0
  last_ml_drop_time <- 0
  last_ml_surv_prob <- 1
}


# Ensure the curve extends horizontally after its last observed event
extended_ml_guided_data <- rbind(
  ml_guided_data,
  data.frame(time = c(last_ml_drop_time, 120), surv = c(last_ml_surv_prob, last_ml_surv_prob))
)
# Remove duplicate points at last_ml_drop_time if they exist due to rbind
extended_ml_guided_data <- extended_ml_guided_data %>% distinct(time, .keep_all = TRUE) %>% arrange(time)


# 10. Styling & annotations
n_total <- nrow(df)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("LT\n(n=%d)", n_total)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) + # Ensure main plot also has these breaks
  scale_y_continuous(
    expand = c(0, 0), limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = seq(0, 100, by = 25)
  ) +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 30, face = "bold"),
    legend.text = element_text(size = 20),
    legend.title = element_blank(),
    legend.key.height = unit(1, "cm"),
    plot.title = element_text(size = 30, face = "bold", hjust = 0.5),
    legend.position = c(1, 0.5),
    legend.justification = c(1, 1),
    plot.margin = unit(c(1,1,1,1), "cm")
  ) +
  guides(color = guide_legend(ncol = 1)) +
  annotate("text", x = 10, y = 0.2, label = p_text, size = 8, hjust = 0) +
  annotate("text", x = 10, y = 0.14, label = hr_text, size = 8, hjust = 0) +
  # Add the extended ML-guided line manually, ensuring it overlays the original KM line
  geom_line(data = extended_ml_guided_data, aes(x = time, y = surv), color = "red", size = 1.2, linetype = "solid")


# The risk table also needs theme adjustments to match the main plot's x-axis text size
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15), # Increase x-axis text size for the risk table
    axis.title.x = element_blank(), # Remove x-axis label from risk table if desired
    axis.ticks.x = element_blank(), # Remove x-axis ticks from risk table if desired
    axis.title.y = element_blank(),  # strata 제목 제거
    plot.title = element_text(size = 20, face = "bold"), # Adjust title size for risk table if present
    plot.margin = unit(c(0, 1,1,1), "cm") # Adjust margin to align with main plot
  )


# 11. Combine plot and risk table
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v", # Align vertically
  rel_heights = c(3, 1) # Adjust these values as needed
)

# 12. Render and Save as high-res PNG
print(combined_final_plot)
ggsave("Figure 4B.eps", plot = combined_final_plot, width = 10, height = 12, dpi = 300)

#################### External_SR #######################

library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot) # For plot_grid

# 1. Load data
file_path <- "C:/Users/andre/Desktop/external_validation_with_survival_columns.xlsx"
df <- read_excel(file_path)

df <- df %>%
  filter(treatment == "surgery")

# 2. Define survival probabilities
df <- df %>%
  mutate(
    old_prob = surv_surgery, # Corrected to surv_surgery since treatment is "surgery"
    new_prob = new_survival_probability
  )

# 3. Fit Cox model
cox_model <- coxph(Surv(survival_time, event) ~ old_prob, data = df)
basehaz_df <- basehaz(cox_model, centered = FALSE)
beta <- coef(cox_model)["old_prob"]
df$lp_new <- beta * df$new_prob

# 4. Predict survival time
get_pred_survtime <- function(lp_val) {
  surv_curve <- exp(-basehaz_df$hazard * exp(lp_val))
  idx <- which(surv_curve <= 0.5)[1]
  if (!is.na(idx)) basehaz_df$time[idx] else max(basehaz_df$time)
}
df$predicted_survival_time <- sapply(df$lp_new, get_pred_survtime)

# 5. Create combined data for KM
combined_data <- data.frame(
  time = c(df$survival_time, pmin(df$predicted_survival_time, 120)),
  event = c(df$event, ifelse(df$predicted_survival_time > 120, 0, 1)), # Event = 0 if predicted beyond 120 (censored)
  group = c(rep("Observed", nrow(df)), rep("ML-guided", nrow(df)))
)
combined_data$group <- factor(combined_data$group, levels = c('Observed',"ML-guided"))

# 6. Fit KM and Cox
km_fit <- survfit(Surv(time, event) ~ group, data = combined_data)
cox_fit <- coxph(Surv(time, event) ~ group, data = combined_data)
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients[,"exp(coef)"]
ci_lower <- summary_cox$conf.int[,"lower .95"]
ci_upper <- summary_cox$conf.int[,"upper .95"]
p_value <- signif(summary_cox$coefficients[,"Pr(>|z|)"], 3)

# 7. Compute mOS
mOS_vals <- surv_median(km_fit)$median
mOS_labels <- ifelse(
  is.na(mOS_vals) | mOS_vals > 120,
  "mOS = u.d.",
  sprintf("mOS = %.2f", mOS_vals)
)

legend_labels <- c(
  sprintf("Real-world treatment %s", mOS_labels[1]),
  sprintf("ML-guided treatment %s", mOS_labels[2])
)

# 8. Custom annotation text
p_text <- if (p_value < 0.001) expression(italic(P) < ".001") else bquote(italic(P) == .(p_value))
hr_text <- sprintf("HR = %.2f (95%% CI: %.2f–%.2f)", hr, ci_lower, ci_upper)

p_combined <- ggsurvplot(
  km_fit,
  data = combined_data,
  conf.int = FALSE,
  pval = FALSE,
  xlab = "Months elapsed",
  ylab = "Probability of survival",
  legend.title = NULL,
  legend.labs = legend_labels,
  ggtheme = theme_classic(),
  xlim = c(0, 120),
  risk.table = TRUE,
  risk.table.y.text = FALSE,
  risk.table.height = 0.25,
  break.time.by = 30,
  palette = c("blue", "red"),
  censor = FALSE,
  surv.scale = "default",
  size = 1.2,
  risk.table.fontsize = 6  # ✅ 추가됨
)

# Extract data for the ML-guided curve from km_fit
ml_guided_data <- with(summary(km_fit, times = seq(0, 120, by = 0.1)), {
  data.frame(
    time = time[strata == "group=ML-guided"],
    surv = surv[strata == "group=ML-guided"]
  )
})

if(nrow(ml_guided_data) > 0){
  last_ml_drop_time <- max(ml_guided_data$time)
  last_ml_surv_prob <- ml_guided_data$surv[ml_guided_data$time == last_ml_drop_time]
} else {
  # If no data points for ML-guided (e.g., all censored at time 0), assume survival 1 at time 0
  last_ml_drop_time <- 0
  last_ml_surv_prob <- 1
}


# Ensure the curve extends horizontally after its last observed event
extended_ml_guided_data <- rbind(
  ml_guided_data,
  data.frame(time = c(last_ml_drop_time, 120), surv = c(last_ml_surv_prob, last_ml_surv_prob))
)
# Remove duplicate points at last_ml_drop_time if they exist due to rbind
extended_ml_guided_data <- extended_ml_guided_data %>% distinct(time, .keep_all = TRUE) %>% arrange(time)


# 10. Styling & annotations
n_total <- nrow(df)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("SR\n(n=%d)", n_total)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) + # Ensure main plot also has these breaks
  scale_y_continuous(
    expand = c(0, 0), limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = seq(0, 100, by = 25)
  ) +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 30, face = "bold"),
    legend.text = element_text(size = 20),
    legend.title = element_blank(),
    legend.key.height = unit(1, "cm"),
    plot.title = element_text(size = 30, face = "bold", hjust = 0.5),
    legend.position = c(1, 0.5),
    legend.justification = c(1, 1),
    plot.margin = unit(c(1,1,1,1), "cm")
  ) +
  guides(color = guide_legend(ncol = 1)) +
  annotate("text", x = 10, y = 0.2, label = p_text, size = 8, hjust = 0) +
  annotate("text", x = 10, y = 0.14, label = hr_text, size = 8, hjust = 0) +
  # Add the extended ML-guided line manually, ensuring it overlays the original KM line
  geom_line(data = extended_ml_guided_data, aes(x = time, y = surv), color = "red", size = 1.2, linetype = "solid")


# The risk table also needs theme adjustments to match the main plot's x-axis text size
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15), # Increase x-axis text size for the risk table
    axis.title.x = element_blank(), # Remove x-axis label from risk table if desired
    axis.ticks.x = element_blank(), # Remove x-axis ticks from risk table if desired
    axis.title.y = element_blank(),  # strata 제목 제거
    plot.title = element_text(size = 20, face = "bold"), # Adjust title size for risk table if present
    plot.margin = unit(c(0, 1,1,1), "cm") # Adjust margin to align with main plot
  )


# 11. Combine plot and risk table
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v", # Align vertically
  rel_heights = c(3, 1) # Adjust these values as needed
)

# 12. Render and Save as high-res PNG
print(combined_final_plot)
ggsave("Figure 4C.eps", plot = combined_final_plot, width = 10, height = 12, dpi = 300)