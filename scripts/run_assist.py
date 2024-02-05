import gi
gi.require_version('Gtk', '3.0')
from gi.repository import gtk

class MainWindow(gtk.Window):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.set_title("Screenshot Example")
        self.set_default_size(1024, 768)
        self.connect("delete-event", self.on_close)

        # 创建截图按钮
        screenshot_button = gtk.Button("Screenshot")
        screenshot_button.connect("clicked", self.on_screenshot)
        self.add(screenshot_button)

        # 创建图像视图用于显示截图
        self.image_view = gtk.Image()
        self.add(self.image_view)

    def on_screenshot(self, button):
        # 获取截图
        screenshot = self.get_snapshot()

        # 将截图转换为 PNG 格式的字节数组
        png_data = screenshot.get_data().decode('utf-8')

        # 创建一个新的 PNG 图像
        png_image = gtk.Image.new_from_data(png_data, gtk.IconSize.DIALOG)

        # 设置图像视图的图像
        self.image_view.set_from_pixbuf(png_image.get_pixbuf())

    def on_close(self, window, event):
        # 关闭窗口时退出应用程序
        gtk.main_quit()

    def get_snapshot(self):
        # 获取整个窗口的截图
        window = self.get_window()
        rect = window.get_bounds()
        screenshot = gtk.gdk.Pixbuf.get_from_window(window, rect.x, rect.y, rect.width, rect.height, gtk.gdk.RGBatsu)
        return screenshot

if __name__ == "__main__":
    # 创建主窗口
    window = MainWindow()
    # 显示主窗口
    window.show_all()
    # 进入主事件循环
    gtk.main()