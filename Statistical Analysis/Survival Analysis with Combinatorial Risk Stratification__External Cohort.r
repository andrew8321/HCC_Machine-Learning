# Load required libraries
library(survival)
library(survminer)
library(readxl)
library(ggpubr)
library(ggplot2)
library(dplyr)
library(cowplot)

# Load the Excel file
file_path <- "C:\\Users\\andre\\Desktop\\HCC연구\\External Validation\\final_data_grouped.xlsx"
sheets <- excel_sheets(file_path)

# ✅ Custom settings
custom_titles <- list(
  "High_LT_High_Surgery" = "Both High-Risk\n(n=149/614, 24.3%)",
  "High_LT_Low_Surgery" = "LT High - SR Low-Risk\n(n=189/614, 30.8%)",
  "Low_LT_High_Surgery" = "LT Low - SR High-Risk\n(n=35/614, 5.7%)",
  "Low_LT_Low_Surgery" = "Both Low-Risk\n(n=238/614, 38.8%)"
)

# ✅ Initialize
plot_list <- list()
median_results <- list()
cox_results <- list()

# ✅ Function with risk table
analyze_survival <- function(sheet_name) {
  df <- read_excel(file_path, sheet = sheet_name)
  
  df$treatment_date <- as.Date(df$treatment_date)
  df$death <- as.Date(df$death)
  df$last_followup <- as.Date(df$last_followup)
  
  df$survival_time <- as.numeric(difftime(pmax(df$death, df$last_followup, na.rm = TRUE), df$treatment_date, units = "days")) / 30.44
  df$event <- ifelse(!is.na(df$death), 1, 0)
  
  df$treatment <- factor(df$treatment)
  df$treatment <- relevel(df$treatment, ref = "surgery")
  
  # Cox model
  cox_model <- coxph(Surv(survival_time, event) ~ treatment, data = df)
  cox_summary <- summary(cox_model)
  hr <- cox_summary$coefficients[,"exp(coef)"]
  ci_lower <- cox_summary$conf.int[,"lower .95"]
  ci_upper <- cox_summary$conf.int[,"upper .95"]
  hr_text_actual <- paste0("HR = ", round(hr, 2), " (95% CI: ", round(ci_lower, 2), "-", round(ci_upper, 2), ")")
  cox_results[[sheet_name]] <<- hr_text_actual
  
  # KM fit
  km_fit <- survfit(Surv(survival_time, event) ~ treatment, data = df)
  
  # Median OS
  median_table <- summary(km_fit)$table
  median_df <- data.frame(
    Treatment = rownames(median_table),
    Median_OS_Months = ifelse(is.na(median_table[, "median"]), "Not Reached", round(median_table[, "median"], 1))
  )
  median_results[[sheet_name]] <<- median_df
  
  # KM plot with risk table
  p_combined <- ggsurvplot(
    km_fit,
    data = df,
    conf.int = FALSE,
    pval = FALSE,
    xlab = "Months elapsed",
    ylab = "Probability of survival",
    legend.title = NULL,
    legend.labs = custom_legends[[sheet_name]],
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
  
  # Style KM plot
  p_combined$plot <- p_combined$plot +
    ggtitle(custom_titles[[sheet_name]]) +
    scale_x_continuous(expand = c(0, 0), limits = c(0, 120)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.25),
                       labels = seq(0, 100, by = 25)) +
    theme(
      axis.text = element_text(size = 20),
      axis.title = element_text(size = 30, face = "bold"),
      legend.text = element_text(size = 20),
      legend.title = element_blank(),
      plot.title = element_text(size = 30, face = "bold", hjust = 0.5),
      legend.key.height = unit(1, "cm"),
      plot.margin = unit(c(1,1,1,1), "cm")
    ) +
    guides(color = guide_legend(ncol = 1)) +
    annotate("text", x = 10, y = 0.2,
         label = paste0("paste(italic('P'), ' = ", p_values[[sheet_name]], "')"),
         size = 8, hjust = 0, parse = TRUE)+
    annotate("text", x = 10, y = 0.14, label = custom_hr_ci[[sheet_name]], size = 8, hjust = 0)
  
  # Set legend position
  legend_pos <- switch(sheet_name,
                       "High_LT_High_Surgery" = c(0.6, 0.75),
                       "High_LT_Low_Surgery" = c(0.85, 0.4),
                       "Low_LT_High_Surgery" = c(0.7, 0.35),
                       "Low_LT_Low_Surgery" = c(0.8, 0.6),
                       c(0,0))
  p_combined$plot <- p_combined$plot + theme(legend.position = legend_pos)
  
  # Style risk table
  p_combined$table <- p_combined$table +
    theme(
      axis.text.x = element_text(size = 15),
      axis.title.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.title.y = element_blank(),
      plot.title = element_text(size = 20, face = "bold"),
      plot.margin = unit(c(0, 1,1,1), "cm")
    )
  
  # Combine plot + table
  combined_final_plot <- plot_grid(
    p_combined$plot,
    p_combined$table,
    ncol = 1,
    align = "v",
    rel_heights = c(3, 1)
  )
  
  return(combined_final_plot)
}

# ✅ Run analysis and save
for (sheet_name in sheets) {
  plot_with_table <- analyze_survival(sheet_name)
  plot_list[[sheet_name]] <- plot_with_table
  
  ggsave(
    filename = paste0(sheet_name, "_KM_with_risktable.png"),
    plot = plot_with_table,
    width = 10, height = 12, dpi = 300, units = "in"
  )
}

# # ✅ Print summaries
# cat("\n\n================ Median OS Summary ================\n")
# for (sheet_name in sheets) {
#   cat("\n[", sheet_name, "]\n")
#   print(median_results[[sheet_name]])
# }

# cat("\n\n================ Cox Model HR (Actual) ================\n")
# for (sheet_name in sheets) {
#   cat("\n[", sheet_name, "] ", cox_results[[sheet_name]], "\n")
# }
