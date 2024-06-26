pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout the repository
                echo 'Checking out the repository...'
                git url: 'https://github.com/Khadijatulkubra001/MLOps-Assignment01.git'
            }
        }

        stage('Code Quality Check') {
            steps {
                // Install required dependencies
                echo 'Installing Flake8...'
                sh 'pip install flake8'
                // Run Flake8 for code quality checks
                echo 'Running Flake8...'
                sh 'flake8 .'
            }
        }

        stage('Build and Push Docker Image') {
            steps {
                // Build Docker image
                echo 'Building Docker image...'
                sh 'docker build -t ahmedbaig137/mlops-assignment01:latest .'
                // Log in to Docker Hub
                echo 'Logging in to Docker Hub...'
                sh 'docker login -u ahmedbaig137 -p Ahmed1282'
                // Push Docker image to Docker Hub
                echo 'Pushing Docker image to Docker Hub...'
                sh 'docker push ahmedbaig137/mlops-assignment01:latest'
            }
        }
    }

    post {
        success {
            // Send email notification upon successful build
            emailext body: 'Jenkins job successfully completed. Docker image pushed to Docker Hub.',
                     subject: 'Jenkins Job Success',
                     to: 'ahmedbaig137@gmail.com'
        }
        failure {
            // Send email notification upon failed build
            emailext body: 'Jenkins job failed. Please check the build logs.',
                     subject: 'Jenkins Job Failure',
                     to: 'ahmedbaig137@gmail.com'
        }
    }
}
