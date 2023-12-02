pipeline {
    agent any

    stages {

        stage('Setup') {
            steps {
                script {
                    // sh "docker stop app_container || true"
                    // sh "docker rm app_container || true"
                    sleep 10
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                // dir('./PycharmProjects') {
                    script {
                        // Run the command to build a Docker image
                        // sh 'docker build -t capstone_app .'
                        sleep 35
                        
                    }
                // }
                
            }
        }

        stage('Run Docker Image') {
            steps {
                // sh 'docker run -d -p 5000:5000 --name app_container app'
                sleep 18

            }
        }

        stage('Email notification'){
            mail bcc: '', body: '''Hi welcome to Jenkins alert messages.
                Thanks,
                Team 24''', cc: '', from: '', replyTo: '', subject: 'Jenkins Job', to: 'nagarjunadoguparthy@gmail.com'
        }

        stage('Wait for Docker Container') {
            steps {
                // Wait for the container to start (you can adjust the sleep time as needed)
                script {
                    sleep 10
                }
            }
        }
    }

    post {
        always {
            // Clean up or perform other actions after the build
            cleanWs()
        }
    }
}