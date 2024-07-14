import os
from typing import Any, Callable, Dict, List


class FolderDatabase:
    """
    folder database 基类，用于管理文件夹中的文件，并提供相关的接口。
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.items: Dict[str, Any] = {}
        self.refresh()

    def filepath(self, filename: str) -> str:
        return os.path.join(self.base_dir, filename)

    def refresh(self):
        self.items = {}
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.base_dir)
                if self.is_valid_file(full_path):
                    data = self.load_item(full_path)
                    if data:
                        self.items[rel_path] = data

        # 检查是否有被删除的项目，同步到 items
        for rel_path in list(self.items.keys()):
            if not os.path.exists(os.path.join(self.base_dir, rel_path)):
                del self.items[rel_path]

    def is_valid_file(self, file_path: str) -> bool:
        # 子类应该重写此方法来定义哪些文件是有效的
        return True

    def load_item(self, file_path: str) -> Any:
        # 子类应该重写此方法来定义如何加载一个项目
        return file_path

    def list_items(self) -> List[Any]:
        return list(self.items.values())

    def create_item(self, item: Any, filename: str):
        rel_path = filename
        full_path = os.path.join(self.base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        self.save_item(item, full_path)
        self.refresh()
        return item

    def save_item(self, item: Any, file_path: str):
        # 子类应该重写此方法来定义如何保存一个项目
        pass

    def get_item(self, get_func: Callable[[Any], bool]) -> Any:
        for item in self.items.values():
            if get_func(item):
                return item
        return None

    def get_item_path(self, get_func: Callable[[Any], bool]) -> str:
        for rel_path, item in self.items.items():
            if get_func(item):
                return rel_path
        return None

    def update_item(self, item: Any, get_func: Callable[[Any], bool]):
        rel_path = self.get_item_path(get_func)
        if rel_path:
            full_path = os.path.join(self.base_dir, rel_path)
            self.save_item(item, full_path)
            self.refresh()
            return item
        else:
            raise ValueError("Item not found for update")

    def save_all(self):
        for rel_path, item in self.items.items():
            full_path = os.path.join(self.base_dir, rel_path)
            self.save_item(item, full_path)

    def __len__(self):
        return len(self.items)
