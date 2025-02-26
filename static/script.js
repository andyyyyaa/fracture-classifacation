document.getElementById('diagnoseButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('请选择一个文件');
        return;
    }

    const loading = document.getElementById('loading');
    loading.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (response.ok) {
            // 使用URL参数传递数据
            const queryParams = new URLSearchParams(data).toString();
            window.location.href = `/result?${queryParams}`;
        } else {
            alert('诊断失败，请重试');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('诊断失败，请重试');
    } finally {
        loading.style.display = 'none';
    }
}); 