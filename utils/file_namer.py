import datetime
import uuid

class FileNamer:
    @staticmethod
    def generate_time_based_name(format_type='default', prefix='', suffix=''):
        """
        生成基于时间的唯一文件夹名称

        Args:
            format_type (str): 命名格式类型
                - 'default': YYYYMMDD_HHMMSS_随机字符
                - 'compact': YYMMDDHHMMSS_随机字符
                - 'readable': YYYY-MM-DD_HH-MM-SS_随机字符
                - 'date_only': YYYYMMDD_随机字符
            prefix (str): 文件夹名称前缀
            suffix (str): 文件夹名称后缀

        Returns:
            str: 生成的唯一文件夹名称
        """
        now = datetime.datetime.now()

        # 生成一个短的随机字符串（取UUID的前8位）
        random_str = str(uuid.uuid4())[:8]

        # 根据不同格式类型生成时间字符串
        if format_type == 'compact':
            time_str = now.strftime('%y%m%d%H%M%S')
        elif format_type == 'readable':
            time_str = now.strftime('%Y-%m-%d_%H-%M-%S')
        elif format_type == 'date_only':
            time_str = now.strftime('%Y%m%d')
        else:  # default
            time_str = now.strftime('%Y%m%d_%H%M%S')

        # 组合文件夹名称
        folder_name = f"{prefix}{'_' if prefix else ''}{time_str}_{random_str}{('_' + suffix) if suffix else ''}"

        return folder_name
    