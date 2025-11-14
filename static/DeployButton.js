import React, { useState } from 'react';

function DeployButton() {
  const [result, setResult] = useState('');

  const handleDeploy = async () => {
    setResult('Deploying...');
    try {
      const response = await fetch('http://localhost:5000/deploy', { method: 'POST' });
      const data = await response.json();
      setResult(data.message || 'Deployment started!');
    } catch (error) {
      setResult('Deployment failed.');
    }
  };

  return (
    <div>
      <button onClick={handleDeploy}>Deploy</button>
      <p>{result}</p>
    </div>
  );
}

export default DeployButton;
