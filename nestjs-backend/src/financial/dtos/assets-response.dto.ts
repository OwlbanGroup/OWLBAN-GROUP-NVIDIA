export class AssetsResponseDto {
  totalAssets: number;
  currency: string;
  assetsByType: AssetByType[];
  assetsByAccount: AssetByAccount[];
  lastUpdated: string;
}

export class AssetByType {
  type: string;
  totalValue: number;
  currency: string;
  accountsCount: number;
  percentage: number;
}

export class AssetByAccount {
  accountId: string;
  accountName: string;
  accountType: string;
  balance: number;
  currency: string;
  lastUpdated: string;
}
