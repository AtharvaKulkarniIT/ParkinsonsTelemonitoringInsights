library(shiny)
library(randomForest)

# Load the saved models
rf_total_lasso <- readRDS("lasso_rf_model_total.rds")
rf_motor_lasso <- readRDS("lasso_rf_model_motor.rds")

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
                    numericInput("age", "Age", value = 0.734693878, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter", "Jitter RAP", value = 0.079287222, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_Abs", "Jitter Abs", value = 0.071164343, min = 0, max = 1, step = 0.01),
                    numericInput("HNR", "HNR", value = 0.085062319, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_APQ11", "Shimmer APQ11", value = 0.039635469, min = 0, max = 1, step = 0.01),
                    numericInput("PPE", "PPE", value = 0.06443344, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_motor", "Jitter", value = 0.0583904800322711, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_DDA_motor", "Shimmer DDA", value = 0.079266526, min = 0, max = 1, step = 0.01)
             )
           )
    ),
    column(width = 6,
           fluidRow(
             column(6,
                    numericInput("Shimmer_total", "Shimmer", value = 0.098029793, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_APQ3_total", "Shimmer APQ3", value = 0.064324419, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_dB_total", "Shimmer dB", value = 0.098029793, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_PPQ5_total", "Jitter PPQ5", value = 0.067543009, min = 0, max = 1, step = 0.01),
                    numericInput("Shimmer_APQ5", "Shimmer APQ5", value = 0.05176393, min = 0, max = 1, step = 0.01),
                    numericInput("DFA", "DFA", value = 0.194543971, min = 0, max = 1, step = 0.01),
                    numericInput("NHR_motor", "NHR", value = 0.018722576, min = 0, max = 1, step = 0.01),
                    numericInput("Jitter_DDP_motor", "Jitter DDP", value = 0.551717473, min = 0, max = 1, step = 0.01)
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
      Jitter.RAP = input$Jitter,
      Jitter.Abs. = input$Jitter_Abs,
      HNR = input$HNR,
      age = input$age,
      Shimmer.APQ11 = input$Shimmer_APQ11,
      PPE = input$PPE,
      Jitter... = input$Jitter_motor,
      Shimmer.DDA = input$Shimmer_DDA_motor,
      NHR = input$NHR_motor,
      Shimmer.APQ5 = input$Shimmer_APQ5,
      DFA = input$DFA,
      Jitter.DDP = input$Jitter_DDP_motor
    )
    
    # Create data frame with input variables for total UPDRS
    new_data_total <- data.frame(
      age = input$age,
      Jitter.RAP = input$Jitter,
      Shimmer = input$Shimmer_total,
      Shimmer.APQ3 = input$Shimmer_APQ3_total,
      Jitter.Abs. = input$Jitter_Abs,
      HNR = input$HNR,
      Shimmer.dB. = input$Shimmer_dB_total,
      Shimmer.APQ11 = input$Shimmer_APQ11,
      PPE = input$PPE,
      Jitter.PPQ5 = input$Jitter_PPQ5_total,
      Shimmer.APQ5 = input$Shimmer_APQ5,
      DFA = input$DFA
    )
    
    # Make predictions
    preds <- predict_UPDRS(new_data_motor, new_data_total)
    
    # Unscaled predicted values
    unscaled_motor_UPDRS <- unscale(preds[[1]], 5, 40 )
    unscaled_total_UPDRS <- unscale(preds[[2]], 7, 52 )
    
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