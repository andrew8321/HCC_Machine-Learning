# Load required libraries
library(survival)
library(survminer)
library(readxl)
library(dplyr)
library(ggplot2)
library(cowplot)

# Load the dataset
file_path <- "C:/Users/andre/Desktop/internal_validation_with_survival_columns.xlsx"
df <- read_excel(file_path)

# Robust tx_date generation
df <- df %>%
  mutate(
    tx_date = case_when(
      treatment == "LT" ~ ifelse(is.na(tx1_name), tx2_date, ifelse(tx1_name == 2, tx1_date, tx2_date)),
      treatment == "surgery" ~ ifelse(is.na(tx1_name), tx2_date, ifelse(tx1_name == 1, tx1_date, tx2_date)),
      TRUE ~ NA_real_
    ),
    tx1_month = ifelse(!is.na(tx_date), tx_date %/% 100 * 12 + tx_date %% 100, NA),
    death_month = ifelse(!is.na(death), death %/% 100 * 12 + death %% 100, NA),
    risk_group = case_when(
      risk_lt == "High Risk" & risk_surgery == "High Risk" ~ "Both High Risk",
      risk_lt == "High Risk" & risk_surgery == "Low Risk" ~ "LT High - SR Low",
      risk_lt == "Low Risk" & risk_surgery == "High Risk" ~ "LT Low - SR High",
      risk_lt == "Low Risk" & risk_surgery == "Low Risk" ~ "Both Low Risk",
      TRUE ~ "Unknown"
    ),
    treatment_type = treatment
  )



# Time variables
study_start_month <- 2008 * 12 + 1
last_followup_month <- 2021 * 12 + 12
analysis_end_month <- 120

# Survival time & event
df <- df %>%
  mutate(
    survival_time_full = case_when(
      !is.na(death_month) & !is.na(tx1_month) & death_month <= last_followup_month ~ death_month - tx1_month,
      !is.na(tx1_month) ~ last_followup_month - tx1_month,
      TRUE ~ NA_real_
    ),
    survival_time = pmin(survival_time_full, analysis_end_month, na.rm = TRUE),
    event = ifelse(!is.na(death_month) & death_month <= last_followup_month & survival_time_full <= analysis_end_month, 1, 0)
  )

# Risk group definitions
risk_groups <- c("Both High Risk", "LT High - SR Low", "LT Low - SR High", "Both Low Risk")

custom_titles <- list(
  "Both High Risk" = "Both High-Risk\n(n=920/3,915, 23.5%)",
  "LT High - SR Low" = "LT High - SR Low-Risk\n(n=1,188/3,915, 30.3%)",
  "LT Low - SR High" = "LT Low - SR High-Risk\n(n=180/3,915, 4.6%)",
  "Both Low Risk" = "Both Low-Risk\n(n=1,627/3,915, 41.6%)"
)

output_filenames <- list(
  "Both High Risk" = "Figure 2A.eps",
  "LT High - SR Low" = "Figure 2B.eps",
  "LT Low - SR High" = "Figure 2C.eps",
  "Both Low Risk" = "Figure 2D.eps"
)

# Initialize
km_plots <- list()
median_os_results <- list()

# Main loop
for (risk in risk_groups) {
  df_subset <- df %>%
    filter(risk_group == risk, !is.na(survival_time), !is.na(event))

  if (nrow(df_subset) > 0) {
    df_subset$treatment <- factor(df_subset$treatment)
    df_subset$treatment <- relevel(df_subset$treatment, ref = "surgery")

    surv_obj <- Surv(time = df_subset$survival_time, event = df_subset$event)
    cox_model <- coxph(surv_obj ~ treatment, data = df_subset)
    cox_summary <- summary(cox_model)

    hr <- round(cox_summary$coefficients[,"exp(coef)"], 2)
    hr_ci_lower <- round(cox_summary$conf.int[,"lower .95"], 2)
    hr_ci_upper <- round(cox_summary$conf.int[,"upper .95"], 2)
    hr_label <- paste0("HR = ", hr, " (95% CI: ", hr_ci_lower, "-", hr_ci_upper, ")")

    km_fit <- survfit(surv_obj ~ treatment, data = df_subset)
    median_table <- summary(km_fit)$table
    median_df <- data.frame(
      Treatment = rownames(median_table),
      Median_OS_Months = ifelse(is.na(median_table[, "median"]), "Not Reached", round(median_table[, "median"], 1))
    )
    median_os_results[[risk]] <- median_df

    km_plot_combined <- ggsurvplot(
      fit = km_fit,
      data = df_subset,
      conf.int = FALSE,
      pval = FALSE,
      xlab = "Months elapsed",
      ylab = "Probability of survival",
      legend.title = NULL,
      legend.labs = custom_legend[[risk]],
      ggtheme = theme_classic(),
      xlim = c(0, 120),
      risk.table = TRUE,
      risk.table.y.text = FALSE,
      risk.table.height = 0.25,
      break.time.by = 30,
      palette = c("red", "blue"),
      censor = FALSE,
      surv.scale = "default",
      size = 1.5,
      risk.table.fontsize = 6
    )

    # Style
    km_plot_combined$plot <- km_plot_combined$plot +
      ggtitle(custom_titles[[risk]]) +
      scale_x_continuous(expand = c(0, 0), limits = c(0, 120), breaks = seq(0, 120, by = 30)) +
      scale_y_continuous(expand = c(0, 0), limits = c(0, 1), breaks = seq(0, 1, by = 0.25), labels = seq(0, 100, by = 25)) +
      theme(
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 30, face = "bold"),
        legend.text = element_text(size = 19),
        legend.title = element_blank(),
        legend.key.height = unit(1, "cm"),
        plot.title = element_text(size = 30, face = "bold", hjust = 0.5),
        legend.position = if (risk == "LT High - SR Low") c(0.9, 0.5) else if (risk == "Both High Risk") c(0.85, 0.8) else c(0.9, 0.65),
        legend.justification = c(1, 1),
        plot.margin = unit(c(1, 1, 1, 1), "cm")
      ) +
        annotate("text", x = 10, y = 0.2,
         label = paste0("paste(italic('P'), ' = ", p_values[[risk]], "')"),
         size = 8, hjust = 0, parse = TRUE)+
      annotate("text", x = 10, y = 0.14, label = hr_label, size = 8, hjust = 0)

    km_plot_combined$table <- km_plot_combined$table +
      theme(
        axis.text.x = element_text(size = 15),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(size = 20, face = "bold"),
        plot.margin = unit(c(0, 1, 1, 1), "cm")
      )

    km_combined <- plot_grid(
      km_plot_combined$plot,
      km_plot_combined$table,
      ncol = 1,
      align = "v",
      rel_heights = c(3, 1)
    )
    km_plots[[risk]] <- km_combined
  }
}

# Save plots
for (risk in risk_groups) {
  if (!is.null(km_plots[[risk]])) {
    ggsave(filename = output_filenames[[risk]],
           plot = km_plots[[risk]],
           width = 10, height = 12, dpi = 300, units = "in")
  }
}

# # Output median OS
# cat("\n\n================ Median OS Summary ================\n")
# for (risk in risk_groups) {
#   if (!is.null(median_os_results[[risk]])) {
#     cat("\n[", risk, "]\n")
#     print(median_os_results[[risk]])
#   }
# }

# # Optional: Debugging output to check for missing data
# cat("\n\n================ Missing Data Summary ================\n")
# df %>%
#   mutate(
#     missing_tx_date = is.na(tx_date),
#     missing_risk_group = risk_group == "Unknown",
#     missing_survival = is.na(survival_time)
#   ) %>%
#   summarise(
#     total = n(),
#     missing_tx_date = sum(missing_tx_date),
#     unknown_risk_group = sum(missing_risk_group),
#     missing_survival_time = sum(missing_survival),
#     used_in_analysis = sum(!missing_tx_date & !missing_risk_group & !missing_survival)
#   ) %>%
#   print()


# df %>%
#   filter(is.na(tx_date)) %>%
#   count(treatment, tx1_name)
