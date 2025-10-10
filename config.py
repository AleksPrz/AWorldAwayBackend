class Config:
    SCHEDULER_TIMEZONE = 'America/Mexico_City'  # Ajusta a tu zona horaria

app.config.from_object(Config)  # sin paréntesis
