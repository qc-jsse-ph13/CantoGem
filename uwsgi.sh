sudo pkill -f uwsgi -9
sleep 1s
rm uwsgi.log
sleep 1s
uwsgi --socket 127.0.0.1:5000 --protocol=http -w wsgi:app --enable-threads --threads 6 --processes 1 --daemonize uwsgi.log
