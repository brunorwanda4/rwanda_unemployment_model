CREATE DATABASE IF NOT EXISTS rwanda_unemployment;
USE rwanda_unemployment;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Sex VARCHAR(50),
    Age INT,
    Marital_status VARCHAR(100),
    Educaional_level VARCHAR(100), -- Match spelling if needed
    hhsize VARCHAR(50),
    TVT2 VARCHAR(100),
    Field_of_education VARCHAR(100),
    Relationship VARCHAR(100),
    predicted_status INT, -- 0 or 1
    probability FLOAT,
    -- Add other relevant fields like user ID if applicable
    INDEX idx_age (Age),
    INDEX idx_edu_level (Educaional_level),
    INDEX idx_pred_status (predicted_status)
);