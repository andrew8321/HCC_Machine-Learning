library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot) # For plot_grid

# 1. Load data
file_path <- "C:/Users/andre/Desktop/external_validation_with_survival_columns_update.xlsx"
df <- read_excel(file_path)

df <- df %>%
  filter(treatment == "LT", LDLT == 1)

# 3. Define survival probabilities
df <- df %>%
  mutate(
    old_prob = surv_lt,  # since treatment == "LT" for all
    new_prob = new_survival_probability
  )

# 4. Fit Cox model using observed probabilities
cox_model <- coxph(Surv(survival_time, event) ~ old_prob, data = df)
basehaz_df <- basehaz(cox_model, centered = FALSE)
beta <- coef(cox_model)["old_prob"]
df$lp_new <- beta * df$new_prob

# 5. Predict survival time from ML-guided risk
get_pred_survtime <- function(lp_val) {
  surv_curve <- exp(-basehaz_df$hazard * exp(lp_val))
  idx <- which(surv_curve <= 0.47)[1]
  if (!is.na(idx)) basehaz_df$time[idx] else max(basehaz_df$time)
}
df$predicted_survival_time <- sapply(df$lp_new, get_pred_survtime)

# 6. Create combined dataset for KM analysis
combined_data <- data.frame(
  time = c(df$survival_time, pmin(df$predicted_survival_time, 120)),
  event = c(df$event, ifelse(df$predicted_survival_time > 120, 0, 1)), # Event = 0 if predicted beyond 120 (censored)
  group = c(rep("Observed", nrow(df)), rep("ML-guided", nrow(df)))
)
combined_data$group <- factor(combined_data$group, levels = c("Observed", "ML-guided"))

# 7. Fit KM and Cox models
km_fit <- survfit(Surv(time, event) ~ group, data = combined_data)
cox_fit <- coxph(Surv(time, event) ~ group, data = combined_data)
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients[,"exp(coef)"]
ci_lower <- summary_cox$conf.int[,"lower .95"]
ci_upper <- summary_cox$conf.int[,"upper .95"]
p_value <- signif(summary_cox$coefficients[,"Pr(>|z|)"], 2)

# 8. Compute mOS for each group
mOS_vals <- surv_median(km_fit)$median
mOS_labels <- ifelse(
  is.na(mOS_vals) | mOS_vals > 120,
  "mOS = u.d.",
  sprintf("mOS = %.1f", mOS_vals)
)

legend_labels <- c(
  sprintf("Real-world DDLT %s", mOS_labels[1]),
  sprintf("ML-guided DDLT %s", mOS_labels[2])
)

# 9. Prepare annotation texts
hr_text <- sprintf("HR = %.2f (95%% CI: %.2f–%.2f)", hr, ci_lower, ci_upper)

# 10. Plot KM curves with risk table
# We will use ggsurvplot for the base and then manually extend the ML-guided curve
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
  censor = FALSE, # Set to FALSE to avoid plotting individual censor marks
  surv.scale = "default",
  size = 1.2,
  risk.table.fontsize = 6 
)

# Extract data for the ML-guided curve from km_fit
ml_guided_data <- with(summary(km_fit, times = seq(0, 120, by = 0.1)), {
  data.frame(
    time = time[strata == "group=ML-guided"],
    surv = surv[strata == "group=ML-guided"]
  )
})

# Find the last time point where survival probability is not 1 (i.e., a drop occurred)
# or the last time point in the observed data if no drop occurred within range.
last_ml_drop_time <- max(ml_guided_data$time)
last_ml_surv_prob <- ml_guided_data$surv[ml_guided_data$time == last_ml_drop_time]

# Ensure the curve extends horizontally after its last observed event
extended_ml_guided_data <- rbind(
  ml_guided_data,
  data.frame(time = c(last_ml_drop_time, 120), surv = c(last_ml_surv_prob, last_ml_surv_prob))
)
# Remove duplicate points at last_ml_drop_time if they exist due to rbind
extended_ml_guided_data <- extended_ml_guided_data %>% distinct(time, .keep_all = TRUE) %>% arrange(time)


# 11. Style the main plot
n_total <- nrow(df)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("DDLT\n(n = %d)", n_total)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) +
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
  # Add the extended ML-guided line manually
  geom_line(data = extended_ml_guided_data, aes(x = time, y = surv), color = "red", size = 1.2, linetype = "solid")


# 12. Style the risk table
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(size = 20, face = "bold"),
    plot.margin = unit(c(0, 1,1,1), "cm")
  )

# 13. Combine and render
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v",
  rel_heights = c(3, 1)
)

print(combined_final_plot)

# # 14. Save high-resolution image
ggsave("Counterfactual_DDLT_external.png", plot = combined_final_plot, width = 10, height = 12, dpi = 300)


################### LDLT #########################
library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot) 
# 1. Load data
file_path <- "C:/Users/andre/Desktop/external_validation_with_survival_columns_update.xlsx"
df <- read_excel(file_path)

df <- df %>%
  filter(treatment == "LT", LDLT == 0)
# 3. Define survival probabilities
df <- df %>%
  mutate(
    old_prob = surv_lt,  # since treatment == "LT" for all
    new_prob = new_survival_probability
  )

# 4. Fit Cox model using observed probabilities
cox_model <- coxph(Surv(survival_time, event) ~ old_prob, data = df)
basehaz_df <- basehaz(cox_model, centered = FALSE)
beta <- coef(cox_model)["old_prob"]
df$lp_new <- beta * df$new_prob

# 5. Predict survival time from ML-guided risk
get_pred_survtime <- function(lp_val) {
  surv_curve <- exp(-basehaz_df$hazard * exp(lp_val))
  idx <- which(surv_curve <= 0.5)[1]
  if (!is.na(idx)) basehaz_df$time[idx] else max(basehaz_df$time)
}
df$predicted_survival_time <- sapply(df$lp_new, get_pred_survtime)

# 6. Create combined dataset for KM analysis
combined_data <- data.frame(
  time = c(df$survival_time, df$predicted_survival_time),
  event = c(df$event, rep(1, nrow(df))),  # assume event=1 for ML-predicted
  group = c(rep("Observed", nrow(df)), rep("ML-guided", nrow(df)))
)
combined_data$group <- factor(combined_data$group, levels = c("Observed", "ML-guided"))

# 7. Fit KM and Cox models
km_fit <- survfit(Surv(time, event) ~ group, data = combined_data)
cox_fit <- coxph(Surv(time, event) ~ group, data = combined_data)
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients[,"exp(coef)"]
ci_lower <- summary_cox$conf.int[,"lower .95"]
ci_upper <- summary_cox$conf.int[,"upper .95"]
p_value <- signif(summary_cox$coefficients[,"Pr(>|z|)"], 3)

# 8. Compute mOS for each group
mOS_vals <- surv_median(km_fit)$median
mOS_labels <- ifelse(
  is.na(mOS_vals) | mOS_vals > 120,
  "mOS = u.d.",
  sprintf("mOS = %.1f", mOS_vals)
)

legend_labels <- c(
  sprintf("Real-world LDLT %s", mOS_labels[1]),
  sprintf("ML-guided LDLT %s", mOS_labels[2])
)
# 9. Prepare annotation texts
p_text <- if (p_value < 0.001) expression(italic(P) < ".001") else bquote(italic(P) == .(p_value))
hr_text <- sprintf("HR = %.2f (95%% CI: %.2f–%.2f)", hr, ci_lower, ci_upper)

# 10. Plot KM curves with risk table
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
  risk.table.fontsize = 6 
)

# 11. Style the main plot
n_total <- nrow(df)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("LDLT\n(n = %d)", n_total)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) +
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
  annotate("text", x = 10, y = 0.14, label = hr_text, size = 8, hjust = 0)

# 12. Style the risk table
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(size = 20, face = "bold"),
    plot.margin = unit(c(0, 1,1,1), "cm")
  )

# 13. Combine and render
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v",
  rel_heights = c(3, 1)
)

print(combined_final_plot)

# # 14. Save high-resolution image
ggsave("Counterfactual_LDLT_external.png", plot = combined_final_plot, width = 10, height = 12, dpi = 300)
