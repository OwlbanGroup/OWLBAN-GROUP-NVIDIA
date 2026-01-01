'use client';

import { useEffect, useState } from 'react';

type Account = {
  id: string;
  name: string;
  type: string;
  currency: string;
};

export default function HomePage() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const orgId = 'ORG_UUID_HERE'; // replace or inject via auth
    fetch(`http://localhost:4000/api/accounts/organization/${orgId}`)
      .then((res) => res.json())
      .then((data) => {
        setAccounts(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <main style={{ padding: 24 }}>
      <h1>Banking Overview</h1>
      <h2>Accounts</h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Currency</th>
          </tr>
        </thead>
        <tbody>
          {accounts.map((acc) => (
            <tr key={acc.id}>
              <td>{acc.name}</td>
              <td>{acc.type}</td>
              <td>{acc.currency}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
