library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot)

# 1. Load data
file_path <- "C:/Users/andre/Desktop/internal_validation_with_survival_columns.xlsx"

df <- read_excel(file_path)


# 포함되지 않는 레벨 포함된 경우 확인
df_surgery %>%
  filter(!(risk_group %in% c("High Risk", "Low Risk")))


# 2. Subset to only surgery group and factorize risk group
df_surgery <- df %>%
  filter(treatment == "surgery") %>%
  mutate(risk_group = factor(risk_surgery, levels = c("High Risk", "Low Risk")))

# 3. Fit KM and Cox models
km_fit <- survfit(Surv(survival_time, event) ~ risk_group, data = df_surgery)
cox_fit <- coxph(Surv(survival_time, event) ~ risk_group, data = df_surgery)

# 4. Median OS
median_df <- surv_median(km_fit)
mos_low <- median_df$median[median_df$strata == "risk_group=Low Risk"]
mos_high <- median_df$median[median_df$strata == "risk_group=High Risk"]

# 5. Cox summary
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients["risk_groupLow Risk", "exp(coef)"]
ci_lower <- summary_cox$conf.int["risk_groupLow Risk", "lower .95"]
ci_upper <- summary_cox$conf.int["risk_groupLow Risk", "upper .95"]
p_value <- summary_cox$coefficients["risk_groupLow Risk", "Pr(>|z|)"]

# 7. Plot
p_combined <- ggsurvplot(
  km_fit,
  data = df_surgery,
  conf.int = FALSE,
  pval = FALSE,
  xlab = "Months elapsed",
  ylab = "Probability of survival",
  legend.title = NULL,
  legend.labs = legend_labels,
  ggtheme = theme_classic(),
  xlim = c(0, 120),
  risk.table = TRUE,
  risk.table.y.text = FALSE, # Keep this TRUE to show the y-axis labels
  risk.table.height = 0.25,
  break.time.by = 30,
  palette = c("red", "blue"),
  censor = FALSE,
  surv.scale = "default",
  size = 1.2,
  risk.table.fontsize = 6 # Font size for the numbers in the risk table
)

# 8. Styling
n_total <- nrow(df_surgery)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("SR\n(n = 3,619)")) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) +
  scale_y_continuous(
    expand = c(0, 0), limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = seq(0, 100, by = 25)
  ) +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 30, face = "bold"),
    legend.text = element_text(size = 19),
    legend.title = element_blank(),
    legend.key.height = unit(1, "cm"),
    plot.title = element_text(size = 30, face = "bold", hjust = 0.5),
    legend.position = c(1, 0.65),
    legend.justification = c(1, 1),
    plot.margin = unit(c(1, 1, 1, 1), "cm")
  ) +
  guides(color = guide_legend(ncol = 1)) +
  annotate("text", x = 10, y = 0.2, label = p_text, size = 8, hjust = 0) +
  annotate("text", x = 10, y = 0.14, label = hr_text, size = 8, hjust = 0)


# Then apply theme modifications
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(), # Remove "strata" title
    axis.ticks.x = element_blank(),
    plot.title = element_text(size = 20, face = "bold"),
    plot.margin = unit(c(0, 1, 1, 1), "cm")
  )


# 10. Combine and save
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v",
  rel_heights = c(3, 1)
)

print(combined_final_plot)
ggsave("KMplot_surgery_risk_group_internal.eps", plot = combined_final_plot, width = 10, height = 12, dpi = 300)


df_surgery %>% filter(is.na(risk_surgery))


################################ LT #################
library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggthemes)
library(grid)
library(cowplot)

# 1. Load data
file_path <- "C:/Users/andre/Desktop/internal_validation_with_survival_columns.xlsx"
df <- read_excel(file_path)

# 2. Subset to only surgery group and factorize risk group
df_surgery <- df %>%
  filter(treatment == "LT") %>%
  mutate(risk_group = factor(risk_lt, levels = c("High Risk", "Low Risk")))  # Low Risk가 위로 오도록 설정

# 3. Fit KM and Cox models
km_fit <- survfit(Surv(survival_time, event) ~ risk_group, data = df_surgery)
cox_fit <- coxph(Surv(survival_time, event) ~ risk_group, data = df_surgery)

# 4. Median OS
median_df <- surv_median(km_fit)
mos_low <- median_df$median[median_df$strata == "risk_group=Low Risk"]
mos_high <- median_df$median[median_df$strata == "risk_group=High Risk"]

# 5. Cox summary
summary_cox <- summary(cox_fit)
hr <- summary_cox$coefficients["risk_groupLow Risk", "exp(coef)"]
ci_lower <- summary_cox$conf.int["risk_groupLow Risk", "lower .95"]
ci_upper <- summary_cox$conf.int["risk_groupLow Risk", "upper .95"]
p_value <- summary_cox$coefficients["risk_groupLow Risk", "Pr(>|z|)"]


# 7. Plot
p_combined <- ggsurvplot(
  km_fit,
  data = df_surgery,
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
  palette = c("red", "blue"),
  censor = FALSE,
  surv.scale = "default",
  size = 1.2,
  risk.table.fontsize = 6
)

# 8. Styling
n_total <- nrow(df_surgery)
p_combined$plot <- p_combined$plot +
  ggtitle(sprintf("LT\n(n = %d)", n_total)) +
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
    plot.margin = unit(c(1, 1, 1, 1), "cm")
  ) +
  guides(color = guide_legend(ncol = 1)) +
  annotate("text", x = 10, y = 0.2, label = p_text, size = 8, hjust = 0) +
  annotate("text", x = 10, y = 0.14, label = hr_text, size = 8, hjust = 0)

# 9. Risk table styling
p_combined$table <- p_combined$table +
  theme(
    axis.text.x = element_text(size = 15),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_blank(), # Remove "strata" title
    plot.title = element_text(size = 20, face = "bold"),
    plot.margin = unit(c(0, 1, 1, 1), "cm")
  )

# 10. Combine and save
combined_final_plot <- plot_grid(
  p_combined$plot,
  p_combined$table,
  ncol = 1,
  align = "v",
  rel_heights = c(3, 1)
)

print(combined_final_plot)
ggsave("KMplot_LT_risk_group_internal.eps", plot = combined_final_plot, width =10, height = 12, dpi = 300)
    