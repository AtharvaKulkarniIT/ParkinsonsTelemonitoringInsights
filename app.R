library(shiny)
library(randomForest)

# Load the saved models
rf_total_lasso <- readRDS("lasso_rf_model_total.rds")
rf_motor_lasso <- readRDS("lasso_rf_model_motor.rds")

# Function for min-max scaling
min_max_scaling <- function(x,maxx,minx) {
  (x - minx) / (maxx - minx)
}

# Unscale function
unscale <- function(x, min_val, max_val) {
  return(x * (max_val - min_val) + min_val)
}

# Define severity interpretation functions
motor_severity_interpretation <- function(score) {
  if (score <= 10) {
    return ("Normal - No motor impairment")
  } else if (score <= 20) {
    return ("Mild impairment - Slight tremor or rigidity, but minimal impact on daily activities")
  } else if (score <= 30) {
    return ("Moderate impairment - Tremor or rigidity is more noticeable and may affect some daily tasks")
  } else if (score <= 40) {
    return ("Severe impairment - Significant tremor, rigidity and slowness that limit daily activities")
  } else if (score <= 50) {
    return ("Very severe impairment - Great difficulty with daily tasks due to severe movement limitations")
  } else {
    return ("Extreme impairment - Requiring constant assistance with movement due to extreme rigidity, tremor and slowness")
  }
}

total_severity_interpretation <- function(score) {
  if (score <= 10) {
    return ("Normal - No or minimal impairment")
  } else if (score <= 20) {
    return ("Early Parkinson's - Very slight symptoms noticeable only on close examination")
  } else if (score <= 30) {
    return ("Mild Parkinson's - Some limitations in daily activities, but generally independent")
  } else if (score <= 40) {
    return ("Moderate Parkinson's - Increased limitations, may require assistance with some daily tasks")
  } else if (score <= 50) {
    return ("Moderately severe Parkinson's - Significant limitations, requiring help with most daily tasks")
  } else {
    return ("Severe Parkinson's - Needing constant assistance with daily activities and mobility")
  }
}

# Define UI
ui <- fluidPage(
  titlePanel("Parkinson's Disease Prediction"),
  fluidRow(
    column(width = 6,
           fluidRow(
             column(6,
                    numericInput("age", "Age", value = 72, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter", "Jitter RAP", value = 0.00401, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_Abs", "Jitter Abs", value = 0.0000338, min = 0, max = 1, step = 0.01),
                    numericInput("HNR", "HNR", value = 21.64, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_APQ11", "Shimmer APQ11", value = 0.01662, min = 0, max = 1, step = 0.01),
                    numericInput("PPE", "PPE", value = 0.16006, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_motor", "Jitter", value = 0.00662, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_DDA_motor", "Shimmer DDA", value = 0.04314, min = 0, max = 1, step = 0.01)
             )
           )
    ),
    column(width = 6,
           fluidRow(
             column(6,
                    numericInput("Shimmer_total", "Shimmer", value = 0.02565, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_APQ3_total", "Shimmer APQ3", value = 0.01438, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_dB_total", "Shimmer dB", value = 0.23, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_PPQ5_total", "Jitter PPQ5", value = 0.00317, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_APQ5", "Shimmer APQ5", value = 0.01309, min = 0, max = 1, step = 0.01),
                    numericInput("DFA", "DFA", value = 0.54842, min = 0, max = 1, step = 0.01),
                    numericInput("NHR_motor", "NHR", value = 0.01429, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_DDP_motor", "Jitter DDP", value = 0.01204, min = 0, max = 1, step = 0.01)
             )
           )
    ),
    column(width = 12,
           actionButton("predictButton", "PREDICT"),
           br(),
           br(),
           h4("Predicted Motor UPDRS:"),
           textOutput("motorUPDRS"),
           h4("Insights:"),
           textOutput("motorUPDRSI"),
           h4("Predicted Total UPDRS:"),
           textOutput("totalUPDRS"),
           h4("Insights:"),
           textOutput("totalUPDRSI"),
           br(),
           br()
    )
  )
)

# Define server logic
server <- function(input, output) {
  # Define predict_UPDRS function outside the observeEvent block
  predict_UPDRS <- function(new_data_motor, new_data_total) {
    # Make predictions using the loaded models (rf_total_lasso and rf_motor_lasso)
    predicted_motor_UPDRS <- predict(rf_motor_lasso, new_data_motor)
    predicted_total_UPDRS <- predict(rf_total_lasso, new_data_total)
    
    # Return the predictions as a list
    return(list(predicted_motor_UPDRS, predicted_total_UPDRS))
  }
  
  # Define function to interpret severity
  interpret_severity <- function(score, motor = FALSE) {
    if (motor) {
      return(motor_severity_interpretation(score))
    } else {
      return(total_severity_interpretation(score))
    }
  }
  
  # Predict when button is clicked
  observeEvent(input$predictButton, {
    # Create data frame with input variables for motor UPDRS
    new_data_motor <- data.frame(
      Jitter.RAP = min_max_scaling(input$Jitter,0.05754,0.00033),
      Jitter.Abs. = min_max_scaling(input$Jitter_Abs,0.00044559,0.00000225),
      HNR = min_max_scaling(input$HNR,37.875,1.659),
      age = min_max_scaling(input$age,85,36),
      Shimmer.APQ11 = min_max_scaling(input$Shimmer_APQ11,0.275466,0.00249),
      PPE = min_max_scaling(input$PPE,0.73173,0.021983),
      Jitter... = min_max_scaling(input$Jitter_motor,0.09999,0.00083),
      Shimmer.DDA = min_max_scaling(input$Shimmer_DDA_motor,0.48802,0.00484),
      NHR = min_max_scaling(input$NHR_motor,0.74826,0.000286),
      Shimmer.APQ5 = min_max_scaling(input$Shimmer_APQ5,0.16702,0.00194),
      DFA = min_max_scaling(input$DFA,0.8656,0.51404),
      Jitter.DDP = min_max_scaling(input$Jitter_DDP_motor,0.17263,0.00098)
    )
    
    # Create data frame with input variables for total UPDRS
    new_data_total <- data.frame(
      NHR = min_max_scaling(input$NHR_motor,0.74826,0.000286),
      Shimmer.DDA = min_max_scaling(input$Shimmer_DDA_motor,0.48802,0.00484),
      
      age = min_max_scaling(input$age,85,36),
      Jitter.RAP = min_max_scaling(input$Jitter,0.05754,0.00033),
      Shimmer = min_max_scaling(input$Shimmer_total,0.26863,0.00306),
      Shimmer.APQ3 = min_max_scaling(input$Shimmer_APQ3_total,0.16267,0.00161),
      Jitter.Abs. = min_max_scaling(input$Jitter_Abs,0.00044559,0.00000225),
      HNR = min_max_scaling(input$HNR,37.875,1.659),
      Shimmer.dB. = min_max_scaling(input$Shimmer_dB_total,2.107,0.026),
      Shimmer.APQ11 = min_max_scaling(input$Shimmer_APQ11,0.275466,0.00249),
      PPE = min_max_scaling(input$PPE,0.73173,0.021983),
      Jitter.PPQ5 = min_max_scaling(input$Jitter_PPQ5_total,0.06956,0.00043),
      Shimmer.APQ5 = min_max_scaling(input$Shimmer_APQ5,0.16702,0.00194),
      DFA = min_max_scaling(input$DFA,0.8656,0.51404)
    )
 #Github - AtharvaKulkarniIT   
    # Make predictions
    preds <- predict_UPDRS(new_data_motor, new_data_total)
    
    # Unscaled predicted values
    unscaled_motor_UPDRS <- unscale(preds[[1]], 5, 40 )
    unscaled_total_UPDRS <- unscale(preds[[2]], 7, 55 )
    
    # Interpret severity
    motor_severity <- interpret_severity(unscaled_motor_UPDRS, motor = TRUE)
    total_severity <- interpret_severity(unscaled_total_UPDRS)
    
    # Output scores and severity interpretations
    output$motorUPDRS <- renderText({
      unscaled_motor_UPDRS
    })
    
    output$totalUPDRS <- renderText({
      unscaled_total_UPDRS
    })
    output$motorUPDRSI <- renderText({
      motor_severity
    })
    
    output$totalUPDRSI <- renderText({
      total_severity
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)
