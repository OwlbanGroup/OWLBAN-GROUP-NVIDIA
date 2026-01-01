type Employee = {
  id: string;
  name: string;
  email: string;
  payRate: string;
  payFrequency: string;
  createdAt: string;
};

type Props = {
  employees: Employee[];
};

export function EmployeeTable({ employees }: Props) {
  if (!employees.length) return <p>No employees yet.</p>;

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <th align="left">Name</th>
          <th align="left">Email</th>
          <th align="right">Pay rate</th>
          <th align="left">Frequency</th>
          <th align="left">Added</th>
        </tr>
      </thead>
      <tbody>
        {employees.map((e) => (
          <tr key={e.id}>
            <td>{e.name}</td>
            <td>{e.email}</td>
            <td align="right">${Number(e.payRate).toFixed(2)}</td>
            <td>{e.payFrequency}</td>
            <td>{new Date(e.createdAt).toLocaleDateString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
