echo "开始清除缓存"
sync;sync;sync
sleep 20
echo 1 > /proc/sys/vm/drop_caches
echo 2 > /proc/sys/vm/drop_caches
echo 3 > /proc/sys/vm/drop_caches
sync
#退出保存并添加权限 
chmod 755 clear_caches.sh

