#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

# Load required libraries
library(shiny)
library(shinydashboard)
library(shinythemes)
library(tidyverse)
library(neuralnet)
library(NeuralNetTools)
library(ggplot2)
library(GGally)
library(caret)
library(writexl)

# Define UI
ui <- dashboardPage(
  dashboardHeader(
    title = "NeuroClassifier"
  ),
  dashboardSidebar(
    sidebarMenu(
      menuItem("About", tabName = "about", icon = icon("info-circle")),
      menuItem("Data Exploration", tabName = "data_exploration", icon = icon("table")),
      menuItem("Analysis", tabName = "analysis", icon = icon("chart-line")),
      fileInput("file", "Upload CSV File", accept = c(".csv")),
      checkboxInput("use_demo_data", "Use Iris Demo Data", value = TRUE),
      selectInput("target", "Select Target (Class) Variable", choices = NULL),
      selectInput("predictors", "Select Predictor Variables", choices = NULL, multiple = TRUE),
      numericInput("hidden_layers", "Number of Hidden Layers", value = 1, min = 1, max = 3),
      uiOutput("neurons_ui"),
      selectInput("act_fct", "Activation Function", choices = c("logistic", "softmax")),
      checkboxInput("linear_output", "Linear Output", value = FALSE),
      numericInput("stepmax", "Maximum Steps for Training", value = 100000, min = 10000, max = 1e6, step = 10000),
      numericInput("threshold", "Threshold for Training", value = 0.01, min = 0.001, max = 0.1, step = 0.001),
      selectInput("validation_split", "Validation Split", choices = c("50:50", "70:30", "80:20"), selected = "70:30"),
      numericInput("learningrate", "Learning Rate", value = 0.01, min = 0.001, max = 0.1, step = 0.001),
      numericInput("repetitions", "Number of Repetitions", value = 1, min = 1, max = 5),
      actionButton("train", "Train Model", class = "btn-primary"),
      downloadButton("download_results", "Download Results as Excel")
    )
  ),
  dashboardBody(
    tabItems(
      # About tab
      tabItem(tabName = "about",
              fluidRow(
                box(
                  title = "NeuroClassifier: A Smart Neural Network Based Classification Platform",
                  width = 12,
                  status = "primary",
                  solidHeader = TRUE,
                  div(
                    style = "padding: 15px;",
                    h4("About NeuroClassifier"),
                    p("NeuroClassifier is a powerful web-based tool designed to make AI-driven neural network classification accessible and practical. This application helps you train, evaluate, and validate classification models using neural networks with ease."),
                    h4("With a user-friendly interface, NeuroClassifier enables users to:"),
                    tags$ul(
                      tags$li(HTML("✅ Upload a dataset or use the demo Iris dataset")),
                      tags$li(HTML("✅ Explore data visually with correlation plots")),
                      tags$li(HTML("✅ Customize the neural network architecture")),
                      tags$li(HTML("✅ Train classification models")),
                      tags$li(HTML("✅ Visualize network architecture")),
                      tags$li(HTML("✅ Evaluate model performance with accuracy metrics")),
                      tags$li(HTML("✅ Download classification results"))
                    ),
                    h4("Contact & Follow"),
                    p("For queries or support, please email: ", tags$a(href = "mailto:imstat09@gmail.com", "imstat09@gmail.com")),
                    h4("Developed By"),
                    p("M. Iqbal Jeelani & Fehim Jeelani (SKUAST-Kashmir)"),
                    p(HTML("📺 Follow us on YouTube: <a href='https://www.youtube.com/@Iqbalstat' target='_blank'>https://www.youtube.com/@Iqbalstat</a>"))
                  )
                )
              )
      ),
      # Data Exploration tab
      tabItem(tabName = "data_exploration",
              fluidRow(
                box(title = "Data Preview", status = "primary", solidHeader = TRUE,
                    dataTableOutput("data_preview"), width = 12)
              ),
              fluidRow(
                box(title = "Correlation Pairs Plot", status = "primary", solidHeader = TRUE,
                    plotOutput("ggpairs_plot", height = "600px"), width = 12)
              )
      ),
      # Analysis tab
      tabItem(tabName = "analysis",
              fluidRow(
                box(title = "Neural Network Summary", verbatimTextOutput("nn_summary"), width = 6),
                box(title = "Network Visualization", plotOutput("plotnet_output"), width = 6)
              ),
              fluidRow(
                box(title = "Raw Network Plot", plotOutput("nn_plot"), width = 12)
              ),
              fluidRow(
                box(title = "Confusion Matrix", verbatimTextOutput("confusion_matrix"), width = 12)
              ),
              fluidRow(
                box(title = "Prediction Results", dataTableOutput("prediction_results"), width = 12)
              )
      )
    )
  )
)

# Server logic
server <- function(input, output, session) {
  
  # Reactive value for the dataset
  data_input <- reactive({
    if(input$use_demo_data) {
      iris
    } else {
      req(input$file)
      tryCatch(
        {
          df <- read.csv(input$file$datapath, header = TRUE, stringsAsFactors = TRUE)
          if(ncol(df) < 2) {
            showNotification("Uploaded file must have at least 2 columns.", type = "error")
            return(NULL)
          }
          df
        },
        error = function(e) {
          showNotification(paste("Error reading the file:", e$message), type = "error")
          return(NULL)
        }
      )
    }
  })
  
  # Update inputs based on dataset
  observeEvent(data_input(), {
    req(data_input())
    all_cols <- names(data_input())
    updateSelectInput(session, "target", choices = all_cols, selected = all_cols[1])
    updateSelectInput(session, "predictors", choices = setdiff(all_cols, input$target), selected = setdiff(all_cols, input$target)[1:min(2, length(setdiff(all_cols, input$target)))])
  })
  
  # Render UI for neurons in hidden layers
  output$neurons_ui <- renderUI({
    req(input$hidden_layers)
    lapply(1:input$hidden_layers, function(i) {
      numericInput(paste0("neurons_", i), paste("Neurons in Layer", i), value = 5, min = 1, max = 20)
    })
  })
  
  # Data preview
  output$data_preview <- renderDataTable({
    req(data_input())
    head(data_input(), 10)
  })
  
  # Generate GGPairs plot
  output$ggpairs_plot <- renderPlot({
    req(data_input(), input$predictors)
    tryCatch({
      df <- data_input()[, c(input$target, input$predictors), drop = FALSE]
      if(ncol(df) > 8) {
        df <- df[, 1:8]
        showNotification("Limiting correlation plot to first 8 variables for visibility", type = "warning")
      }
      if(nrow(df) > 500) df <- df[sample(nrow(df), 500), ]
      if(is.factor(df[[input$target]]) || is.character(df[[input$target]])) {
        if(is.character(df[[input$target]])) df[[input$target]] <- as.factor(df[[input$target]])
        ggpairs(df, title = "Data Correlation Plot", mapping = aes(colour = .data[[input$target]]),
                lower = list(combo = wrap("facethist", binwidth = 1)))
      } else {
        ggpairs(df, title = "Data Correlation Plot")
      }
    }, error = function(e) {
      plot(1, 1, main = "Error creating correlation plot", sub = paste("Error:", e$message),
           type = "n", xlab = "", ylab = "")
    })
  })
  
  # Preprocess data function
  preprocess_data <- reactive({
    req(data_input(), input$target, input$predictors)
    df <- data_input()[, c(input$target, input$predictors), drop = FALSE]
    df <- na.omit(df)
    if(nrow(df) < 10) {
      showNotification("Not enough valid data after filtering (minimum 10 rows required).", type = "error")
      return(NULL)
    }
    if(is.factor(df[[input$target]]) || is.character(df[[input$target]])) {
      if(length(unique(df[[input$target]])) < 2) {
        showNotification("Target variable must have at least 2 unique classes for classification.", type = "error")
        return(NULL)
      }
    }
    
    target_var <- df[[input$target]]
    norm_func <- function(x) {
      if(is.numeric(x) && (max(x, na.rm = TRUE) - min(x, na.rm = TRUE) > 0))
        (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)) else x
    }
    
    df_norm <- df
    for(col in input$predictors) if(is.numeric(df[[col]])) df_norm[[col]] <- norm_func(df[[col]])
    
    if(is.factor(target_var) || is.character(target_var)) {
      if(is.character(target_var)) target_var <- as.factor(target_var)
      df_norm[[input$target]] <- NULL
      for(level in levels(target_var)) df_norm[[level]] <- ifelse(target_var == level, 1, 0)
      attr(df_norm, "target_levels") <- levels(target_var)
      attr(df_norm, "original_target") <- target_var
    } else {
      df_norm[[input$target]] <- norm_func(target_var)
    }
    return(df_norm)
  })
  
  # Train-test split
  train_test_split <- reactive({
    req(preprocess_data())
    split_ratio <- as.numeric(strsplit(input$validation_split, ":")[[1]]) / 100
    set.seed(123)
    sample_size <- floor(split_ratio[1] * nrow(preprocess_data()))
    train_indices <- sample(seq_len(nrow(preprocess_data())), size = sample_size)
    list(trainset = preprocess_data()[train_indices, ], testset = preprocess_data()[-train_indices, ],
         test_indices = which(!(seq_len(nrow(preprocess_data())) %in% train_indices)))
  })
  
  # Reactive values for model and predictions
  model_store <- reactiveVal(NULL)
  predictions_store <- reactiveVal(NULL)
  
  # Train model when button is clicked
  observeEvent(input$train, {
    req(train_test_split())
    target_levels <- attr(preprocess_data(), "target_levels")
    
    if(is.null(target_levels)) {
      formula_str <- paste(input$target, "~", paste(input$predictors, collapse = " + "))
    } else {
      formula_str <- paste(paste(attr(preprocess_data(), "target_levels"), collapse = " + "), "~", 
                           paste(input$predictors, collapse = " + "))
    }
    formula <- as.formula(formula_str)
    
    hidden_layers <- numeric(0)
    for(i in 1:input$hidden_layers) {
      if(!is.null(input[[paste0("neurons_", i)]])) hidden_layers <- c(hidden_layers, input[[paste0("neurons_", i)]])
    }
    if(length(hidden_layers) == 0) hidden_layers <- 5
    
    showNotification("Training neural network...", type = "message", duration = NULL, id = "training")
    
    model <- tryCatch({
      nn <- neuralnet(formula, data = train_test_split()$trainset, hidden = hidden_layers,
                      rep = input$repetitions, act.fct = "logistic", err.fct = "sse", # Using logistic as base activation
                      linear.output = input$linear_output, lifesign = "minimal",
                      stepmax = input$stepmax, threshold = input$threshold,
                      learningrate = input$learningrate)
      removeNotification(id = "training")
      showNotification("Training complete!", type = "message")
      nn
    }, error = function(e) {
      removeNotification(id = "training")
      showNotification(paste("Error in training:", e$message), type = "error")
      NULL
    })
    
    model_store(model)
    
    if(!is.null(model)) {
      test_data <- train_test_split()$testset[, input$predictors, drop = FALSE]
      predictions <- compute(model, test_data)
      
      if(!is.null(target_levels)) {
        idx <- apply(predictions$net.result, 1, which.max)
        predicted_class <- factor(target_levels[idx], levels = target_levels)
        actual_class <- attr(preprocess_data(), "original_target")[train_test_split()$test_indices]
        results <- data.frame(Actual = actual_class, Predicted = predicted_class)
        for(i in seq_along(target_levels)) results[[paste0("Prob_", target_levels[i])]] <- predictions$net.result[, i]
      } else {
        results <- data.frame(Actual = train_test_split()$testset[[input$target]], 
                              Predicted = predictions$net.result)
      }
      predictions_store(results)
    }
  })
  
  # Render neural network summary
  output$nn_summary <- renderPrint({
    req(model_store())
    tryCatch({
      print(model_store())
      cat("\nModel Details:\n")
      cat(paste("Activation Function:", "logistic", "\n")) # Fixed to logistic, as softmax is not natively supported
      cat(paste("Error Function: sse\n"))
      cat(paste("Learning Rate:", input$learningrate, "\n"))
      hidden_layers <- numeric(0)
      for(i in 1:input$hidden_layers) if(!is.null(input[[paste0("neurons_", i)]])) hidden_layers <- c(hidden_layers, input[[paste0("neurons_", i)]])
      cat(paste("Hidden Layers:", paste(hidden_layers, collapse = "-"), "\n"))
      if(!is.null(model_store()$result.matrix)) {
        if(is.null(dim(model_store()$result.matrix))) {
          cat("\nTraining Info:\n")
          cat(paste("Steps:", model_store()$result.matrix["steps"], "\n"))
          cat(paste("Error:", round(model_store()$result.matrix["error"], 4), "\n"))
        } else {
          best_idx <- which.min(model_store()$result.matrix["error", ])
          cat("\nTraining Info (Best Network):\n")
          cat(paste("Repetition:", best_idx, "\n"))
          cat(paste("Steps:", model_store()$result.matrix["steps", best_idx], "\n"))
          cat(paste("Error:", round(model_store()$result.matrix["error", best_idx], 4), "\n"))
        }
      }
    }, error = function(e) {
      cat("Error generating summary:", e$message, "\n")
    })
  })
  
  # Render neural network plot
  output$nn_plot <- renderPlot({
    req(model_store())
    tryCatch({
      best_rep <- if(!is.null(model_store()$result.matrix) && ncol(model_store()$result.matrix) > 1) 
        which.min(model_store()$result.matrix["error", ]) else 1
      plot(model_store(), rep = best_rep, main = "Raw Neural Network Plot")
    }, error = function(e) {
      plot(1, type = "n", xlab = "", ylab = "", main = paste("Error generating plot:", e$message))
    })
  })
  
  # Render plotnet visualization
  output$plotnet_output <- renderPlot({
    req(model_store())
    tryCatch({
      best_rep <- if(!is.null(model_store()$result.matrix) && ncol(model_store()$result.matrix) > 1) 
        which.min(model_store()$result.matrix["error", ]) else 1
      plotnet(model_store(), rep = best_rep, alpha.val = 0.8, circle_col = c('purple', 'white', 'white'),
              bord_col = 'black', main = "Network Visualization")
    }, error = function(e) {
      plot(1, type = "n", xlab = "", ylab = "", main = paste("Error generating plotnet:", e$message))
    })
  })
  
  # Render confusion matrix
  output$confusion_matrix <- renderPrint({
    req(predictions_store())
    if(is.factor(predictions_store()$Actual)) {
      cm <- confusionMatrix(predictions_store()$Predicted, predictions_store()$Actual)
      # Custom print to exclude Mcnemar's Test
      cat("Confusion Matrix and Statistics\n\n")
      print(cm$table)
      cat("\nOverall Statistics\n\n")
      cat(paste("               Accuracy : ", round(cm$overall["Accuracy"], 4), "\n", sep = ""))
      cat(paste("                 95% CI : (", round(cm$overall["AccuracyLower"], 4), ", ", round(cm$overall["AccuracyUpper"], 4), ")\n", sep = ""))
      cat(paste("    No Information Rate : ", round(cm$overall["AccuracyNull"], 4), "\n", sep = ""))
      cat(paste("    P-Value [Acc > NIR] : ", format(cm$overall["AccuracyPValue"], scientific = TRUE), "\n", sep = ""))
      cat(paste("                  Kappa : ", round(cm$overall["Kappa"], 4), "\n", sep = ""))
      cat("\nStatistics by Class:\n\n")
      print(cm$byClass)
    } else {
      actual <- predictions_store()$Actual
      predicted <- predictions_store()$Predicted
      rmse <- sqrt(mean((predicted - actual)^2))
      r2 <- cor(predicted, actual)^2
      mae <- mean(abs(predicted - actual))
      cat("Regression Metrics:\n")
      cat(paste("R-squared:", round(r2, 4), "\n"))
      cat(paste("RMSE:", round(rmse, 4), "\n"))
      cat(paste("MAE:", round(mae, 4), "\n"))
    }
  })
  
  # Render prediction results table
  output$prediction_results <- renderDataTable({
    req(predictions_store())
    predictions_store()
  })
  
  # Download results handler
  output$download_results <- downloadHandler(
    filename = function() paste("neural_network_results_", Sys.Date(), ".xlsx", sep = ""),
    content = function(file) {
      req(predictions_store())
      write_xlsx(list("Prediction Results" = predictions_store()), path = file)
    }
  )
}

# Run the Shiny App
shinyApp(ui = ui, server = server)