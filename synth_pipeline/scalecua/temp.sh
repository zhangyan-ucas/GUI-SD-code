# 1. 设定你的 Root 文件夹路径并进入该目录
ROOT_DIR="/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/data/data_20250526"
cd "$ROOT_DIR" || exit

# 2. 开始循环处理
for prefix in windows ; do
    echo "📦 正在合并并解压: $prefix ..."
    
    # 合并分卷文件
    cat ${prefix}.tar.gz.part-* > ${prefix}.tar.gz
    
    # 解压合并后的文件
    tar -xzf ${prefix}.tar.gz
    
    echo "✅ $prefix 处理完成！"
    echo "------------------------"
done

echo "🎉 所有文件处理完毕！"