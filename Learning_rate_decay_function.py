def learning_rate_decay_function(learning_rate, episode):
    return learning_rate #/ (1 + episode * 0.0005)