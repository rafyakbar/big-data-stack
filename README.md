# Upgrade DB
docker exec -it superset superset db upgrade

# Buat user admin (jika belum ada)
docker exec -it superset superset fab create-admin --username admin --firstname Admin --lastname User --email admin@example.com --password admin

# Atau reset password jika user sudah ada
# docker exec -it superset superset fab reset-password --username admin --password admin

# Init
docker exec -it superset superset init