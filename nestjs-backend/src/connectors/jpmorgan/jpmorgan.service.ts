import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';

interface JpmAccount {
  id: string;
  name: string;
  type: string;
  currency: string;
}

interface JpmBalance {
  accountId: string;
  available: string;
  current: string;
  asOf: string;
}

interface JpmTransaction {
  id: string;
  accountId: string;
  amount: string;
  currency: string;
  direction: 'CREDIT' | 'DEBIT';
  description: string;
  merchantName?: string;
  category?: string;
  postedAt: string;
}

@Injectable()
export class JpmorganService {
  private readonly logger = new Logger(JpmorganService.name);
  private readonly baseUrl = process.env.JPM_API_BASE_URL;
  private readonly clientId = process.env.JPM_CLIENT_ID;
  private readonly clientSecret = process.env.JPM_CLIENT_SECRET;

  constructor(private readonly http: HttpService) {}

  // TODO: real token flow; this is just a placeholder
  private async getAccessToken(): Promise<string> {
    // Use client credentials or OAuth depending on JPM setup
    return 'ACCESS_TOKEN';
  }

  async fetchAccounts(connectionRef: string): Promise<JpmAccount[]> {
    const token = await this.getAccessToken();
    const url = `${this.baseUrl}/connections/${connectionRef}/accounts`;

    const res = await this.http
      .get<JpmAccount[]>(url, {
        headers: {
          Authorization: `Bearer ${token}`,
          'X-Client-Id': this.clientId,
        },
      })
      .toPromise();

    return res.data;
  }

  async fetchBalances(connectionRef: string): Promise<JpmBalance[]> {
    const token = await this.getAccessToken();
    const url = `${this.baseUrl}/connections/${connectionRef}/balances`;

    const res = await this.http
      .get<JpmBalance[]>(url, {
        headers: { Authorization: `Bearer ${token}` },
      })
      .toPromise();

    return res.data;
  }

  async fetchTransactions(
    connectionRef: string,
    params: { fromDate?: string; toDate?: string },
  ): Promise<JpmTransaction[]> {
    const token = await this.getAccessToken();
    const url = `${this.baseUrl}/connections/${connectionRef}/transactions`;

    const res = await this.http
      .get<JpmTransaction[]>(url, {
        headers: { Authorization: `Bearer ${token}` },
        params,
      })
      .toPromise();

    return res.data;
  }
}
