pipeline {
  agent any
  options {
    buildDiscarder(logRotator(numToKeepStr: '5'))
  }
  environment {
    DOCKERHUB_CREDENTIALS = credentials('tomdockerhub')
  }
  stages {
    stage('Build') {
      steps {
        sh 'docker build -t oli2/ludwig .'
      }
    }
    stage('Login') {
      steps {
        sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
      }
    }
    stage('Push') {
      steps {
        sh 'docker push oli2/ludwig'
      }
    }
  }
  post {
    always {
      sh 'docker logout'
    }
  }
}
