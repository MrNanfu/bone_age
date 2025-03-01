import React, { useState } from "react";
import { Upload, Button, Input, Table, Card, Layout, Typography, Form, notification } from "antd";
import { UploadOutlined } from "@ant-design/icons";

const { Header, Content } = Layout;
const { Title } = Typography;

const GrowthAssistant = () => {
  const [boneAge, setBoneAge] = useState(null);
  const [physiqueData, setPhysiqueData] = useState({ height: "", weight: "", bmi: "" });
  const [history, setHistory] = useState([]);
  
  const handleUpload = (info) => {
    if (info.file.status === "done") {
      setBoneAge(Math.floor(Math.random() * 10) + 5); // Mock prediction result
      notification.success({ message: "X 光片上传成功，骨龄分析完成！" });
    }
  };
  
  const handlePhysiqueSubmit = () => {
    const newEntry = { ...physiqueData, result: "正常" };
    setHistory([...history, newEntry]);
    notification.success({ message: "体格评估完成！" });
  };

  const columns = [
    { title: "身高 (cm)", dataIndex: "height", key: "height" },
    { title: "体重 (kg)", dataIndex: "weight", key: "weight" },
    { title: "BMI", dataIndex: "bmi", key: "bmi" },
    { title: "评估结果", dataIndex: "result", key: "result" }
  ];

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header style={{ background: "#1890ff", color: "#fff", textAlign: "center", fontSize: "24px" }}>
        智能成长助手
      </Header>
      <Content style={{ padding: "20px" }}>
        <Card title="上传 X 光片进行骨龄预测">
          <Upload action="/api/upload" showUploadList={false} onChange={handleUpload}>
            <Button icon={<UploadOutlined />}>点击上传</Button>
          </Upload>
          {boneAge && <p>预测骨龄：{boneAge} 岁</p>}
        </Card>

        <Card title="输入体格数据进行评估" style={{ marginTop: 20 }}>
          <Form layout="vertical" onFinish={handlePhysiqueSubmit}>
            <Form.Item label="身高 (cm)" required>
              <Input placeholder="请输入身高" onChange={(e) => setPhysiqueData({ ...physiqueData, height: e.target.value })} />
            </Form.Item>
            <Form.Item label="体重 (kg)" required>
              <Input placeholder="请输入体重" onChange={(e) => setPhysiqueData({ ...physiqueData, weight: e.target.value })} />
            </Form.Item>
            <Form.Item label="BMI" required>
              <Input placeholder="请输入 BMI" onChange={(e) => setPhysiqueData({ ...physiqueData, bmi: e.target.value })} />
            </Form.Item>
            <Button type="primary" htmlType="submit">提交评估</Button>
          </Form>
        </Card>

        <Card title="历史记录" style={{ marginTop: 20 }}>
          <Table dataSource={history} columns={columns} rowKey="height" />
        </Card>
      </Content>
    </Layout>
  );
};

export default GrowthAssistant;
